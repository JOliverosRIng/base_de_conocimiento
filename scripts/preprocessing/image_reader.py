"""
image_reader.py
===============

Lectura de imágenes por OCR para CODEFEST AD ASTRA 2026. Se encarga
ÚNICAMENTE de extraer el texto que contenga una imagen (infografías,
gráficos, tablas, mapas con etiquetas) mediante OCR clásico (Tesseract),
que la Sección 2.1 del documento guía recomienda explícitamente.

El procesamiento del texto (limpieza, chunking, validación) NO vive aquí:
está en text_processor.py, reutilizable para cualquier fuente.

Uso básico
----------
    from image_reader import ImageReader

    reader = ImageReader()
    resultado = reader.extraer("infografia.png")
    # resultado = {"texto": "...", "titulo": "", "fecha": "",
    #              "confianza": 78.5, "tiene_texto": True}

    # O el flujo completo a JSONL:
    from image_reader import process_image
    jsonl = process_image("infografia.png", fenomeno=1)

Notas
-----
Muchas imágenes (retratos, fotos, portadas) NO contienen texto útil. En
esos casos Tesseract devuelve vacío o "basura" de baja confianza, y esta
clase la descarta automáticamente mediante el filtro de confianza.

Dependencias:
    pip install pytesseract pillow
    # Binario del sistema (motor OCR):
    #   Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-por
    #   Windows: instalador de https://github.com/UB-Mannheim/tesseract/wiki
"""

import pytesseract
from PIL import Image, ImageOps, ImageFilter


class ImageReader:
    """
    Lee una imagen y devuelve el texto que contenga, vía OCR.

    Aplica preprocesamiento (escala de grises, autocontraste, nitidez) para
    mejorar el reconocimiento en infografías y gráficos, y filtra por
    confianza para descartar el "texto basura" que Tesseract produce cuando
    la imagen en realidad no tiene texto.

    Parámetros del constructor
    ---------------------------
    ocr_idiomas : str
        Idiomas que Tesseract intentará reconocer (por defecto es+en+pt).
    min_confianza : float
        Confianza promedio mínima (0-100) para aceptar el texto extraído.
        Por debajo de este valor se considera basura y se descarta.
    min_caracteres : int
        Mínimo de caracteres para considerar que hay texto útil.
    preprocesar : bool
        Si es True, aplica grises + autocontraste + nitidez antes del OCR.
    escala : float
        Factor de ampliación de la imagen antes del OCR. Ampliar ayuda con
        texto pequeño (típico en gráficos). 1.0 = sin ampliar.
    """

    FORMATOS_SOPORTADOS = (".png", ".jpg", ".jpeg", ".tif", ".tiff",
                           ".bmp", ".gif", ".webp")

    def __init__(
        self,
        ocr_idiomas: str = "spa+eng+por",
        min_confianza: float = 40.0,
        min_caracteres: int = 10,
        preprocesar: bool = True,
        escala: float = 2.0,
    ):
        self.ocr_idiomas = ocr_idiomas
        self.min_confianza = min_confianza
        self.min_caracteres = min_caracteres
        self.preprocesar = preprocesar
        self.escala = escala

    # ------------------------------------------------------------------
    # Preprocesamiento de la imagen
    # ------------------------------------------------------------------
    def _preprocesar_imagen(self, imagen: Image.Image) -> Image.Image:
        """Mejora la imagen para el OCR:
          - Convierte a escala de grises (elimina ruido de color).
          - Aplica autocontraste (separa mejor el texto del fondo).
          - Aumenta la nitidez (bordes de letra más definidos).
          - Amplía la imagen (ayuda con texto pequeño de gráficos).
        Devuelve la imagen procesada; no modifica la original."""
        img = imagen.convert("L")                 # escala de grises
        img = ImageOps.autocontrast(img)          # autocontraste
        img = img.filter(ImageFilter.SHARPEN)     # nitidez

        if self.escala and self.escala != 1.0:
            nuevo_tam = (int(img.width * self.escala),
                         int(img.height * self.escala))
            img = img.resize(nuevo_tam, Image.LANCZOS)

        return img

    # ------------------------------------------------------------------
    # OCR con filtro de confianza
    # ------------------------------------------------------------------
    def _ocr_con_confianza(self, imagen: Image.Image) -> tuple[str, float]:
        """Ejecuta OCR y devuelve (texto, confianza_promedio).

        Tesseract no sabe de antemano si la imagen tiene texto: siempre lo
        intenta. La confianza promedio de las palabras reconocidas es la
        señal para distinguir texto real de basura."""
        try:
            datos = pytesseract.image_to_data(
                imagen, lang=self.ocr_idiomas,
                output_type=pytesseract.Output.DICT
            )
        except Exception:
            # Si el OCR falla (idioma no instalado, binario ausente, etc.)
            # no se rompe el flujo: se reporta como sin texto.
            return ("", 0.0)

        palabras = []
        confianzas = []
        for texto_palabra, conf in zip(datos.get("text", []),
                                       datos.get("conf", [])):
            try:
                conf_num = float(conf)
            except (TypeError, ValueError):
                continue
            # Tesseract usa -1 para entradas sin palabra reconocida.
            if conf_num < 0:
                continue
            if texto_palabra and texto_palabra.strip():
                palabras.append(texto_palabra.strip())
                confianzas.append(conf_num)

        texto = " ".join(palabras).strip()
        confianza_prom = (sum(confianzas) / len(confianzas)) if confianzas else 0.0
        return (texto, confianza_prom)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def extraer(self, ruta_imagen: str) -> dict:
        """Lee una imagen y devuelve el texto detectado por OCR.

        Si el texto no supera los umbrales de confianza y longitud, se
        considera que la imagen NO tiene texto útil y se devuelve texto
        vacío (la imagen se descarta automáticamente).

        Retorna
        -------
        dict
            {"texto": str, "titulo": str, "fecha": str,
             "confianza": float, "tiene_texto": bool}
        """
        try:
            imagen = Image.open(ruta_imagen)
        except Exception:
            return {"texto": "", "titulo": "", "fecha": "",
                    "confianza": 0.0, "tiene_texto": False}

        if self.preprocesar:
            imagen = self._preprocesar_imagen(imagen)

        texto, confianza = self._ocr_con_confianza(imagen)

        # Filtro: descarta la "basura" que Tesseract produce en imágenes
        # sin texto real (retratos, fotos, ilustraciones).
        tiene_texto = (
            len(texto) >= self.min_caracteres
            and confianza >= self.min_confianza
        )
        if not tiene_texto:
            texto = ""

        return {
            "texto": texto,
            "titulo": "",      # las imágenes no traen título interno
            "fecha": "",       # ni fecha interna
            "confianza": round(confianza, 2),
            "tiene_texto": tiene_texto,
        }


# ----------------------------------------------------------------------
# Función orquestadora: process_image (une ImageReader + TextProcessor)
# ----------------------------------------------------------------------
def process_image(image_path: str, fenomeno: int,
                  encoder_name: str = "intfloat/multilingual-e5-base",
                  imprimir_validacion: bool = False,
                  ocr_idiomas: str = "spa+eng+por",
                  min_confianza: float = 40.0) -> str:
    """
    Procesa una imagen por OCR y devuelve el JSONL con sus chunks.
    Orquesta la lectura (ImageReader) y el procesamiento (TextProcessor).

    Si la imagen no contiene texto útil (según el filtro de confianza),
    devuelve una cadena VACÍA: no hay nada que indexar.

    Parámetros
    ----------
    image_path : str
        Ruta a la imagen a procesar.
    fenomeno : int
        Fenómeno temático (1, 2 o 3).
    encoder_name : str, opcional
        Encoder cuyo tokenizer se usa para contar tokens.
    imprimir_validacion : bool, opcional
        Imprime los problemas de validación si los hay.
    ocr_idiomas : str, opcional
        Idiomas para Tesseract.
    min_confianza : float, opcional
        Confianza mínima para aceptar el texto del OCR.

    Retorna
    -------
    str
        JSON Lines (un chunk por línea), o cadena vacía si no hubo texto.
    """
    from text_processor import TextProcessor

    reader = ImageReader(ocr_idiomas=ocr_idiomas, min_confianza=min_confianza)
    datos = reader.extraer(image_path)

    # Sin texto útil: no se genera ningún chunk.
    if not datos["tiene_texto"] or not datos["texto"].strip():
        return ""

    procesador = TextProcessor(
        encoder_name=encoder_name,
        imprimir_validacion=imprimir_validacion,
    )
    return procesador.procesar_a_jsonl(
        texto=datos["texto"],
        fuente=image_path,
        fenomeno=fenomeno,
        formato="imagen",
        titulo=datos["titulo"],
        fecha=datos["fecha"],
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        salida = process_image(sys.argv[1], int(sys.argv[2]))
        print(salida if salida else "(La imagen no contiene texto útil)")
    else:
        print("Uso: python image_reader.py <ruta_imagen> <fenomeno>")