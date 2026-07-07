

Hay distintos modelos para generar embedings:
* No puedes usar a mitad de vida del proyecto cambiar el modelo LLM para crear embedings, sino deberias reindexar todos lso documentos


Strategias chunking:
* Mecanica:
  - Nunca usar FixedSized (cada ciertos carácteres o palabras).
  - Recursive: Indicar un tamaño estimado y encontrar separadores como "." o "\n" y calcular token incrementalmente hasta llegar al estimado.
  - Coste = 0€
TIPS: 
   - Añadir overlap entre 10% y 20% (añadir parte del chunk anterior al siguiente) para mejorar la calidad de la búsqueda semántica. Esto ayuda a que los chunks tengan contexto compartido y no se pierda información relevante entre ellos.
   - Revisar qué chunks son inútiles (por tamaño o por contenido)


* Estructural:
    - Usar una estructura del chunck
    - Usar una jerarquía de chunks
    - Coste = 0€

* Semántico:
    - Usar un modelo LLM para generar chunks semánticos
    - Coste = 0.0001€ por chunk (aprox) -> riesgo medir ROI

* Contextual o proposicional:
    - LateChunking: Usar un modelo LLM para generar chunks en base a todo el documento, basados en contexto y luego hace represenataciones internas
    - Retrieval-Contextual:
      -- Antes de convertir el embeding le inyecto al chunk información de contexto adicional (metadatos, resumen, etc.) para mejorar la calidad de la búsqueda semántica.
      -- Es como un QueryExpansion pero a nivel de chunk
      -- Ejemplo: titulo, ruta jerárquica, resumen, tipo de documento, dominio funcional, etc.
    - Coste = Más caro que los anteriores-> riesgo medir ROI

TIP: Información que quieres que se comprenda extiende el chunk, información a filtrar en metadatos
TIP: Para filtrar palabras es mejor el algoritmo BM25 que un LLM, ya que el LLM no es determinista y puede fallar en la búsqueda de palabras exactas. BM25 es más eficiente para búsquedas exactas y filtrado de palabras clave.