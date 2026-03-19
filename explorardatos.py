"""
LegalGuard RAG — Exploración de Synthetic Q&A y PDF WHO SOP
Ejecutar: python explorar_datos.py

Analiza:
  1. Synthetic Q&A  (data/synthetic_qa/synthetic_qa_train.jsonl)
  2. PDF WHO SOP    (data/who_sop/who_mch_clinic_sop.pdf)
"""

import json
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC Q&A
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("💬 PARTE 1 — Synthetic Q&A Dataset")
print("=" * 65)

RUTA_JSONL = Path("data/synthetic_qa/synthetic_qa_train.jsonl")

if not RUTA_JSONL.exists():
    print(f"\n❌ No se encontró: {RUTA_JSONL}")
    print("   Asegúrate de haber ejecutado descargar_synthetic.py")
else:
    # Cargar todas las líneas
    ejemplos = []
    with open(RUTA_JSONL, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    ejemplos.append(json.loads(linea))
                except json.JSONDecodeError:
                    pass

    print(f"\n✅ Total de ejemplos cargados : {len(ejemplos)}")
    print(f"   Tamaño del archivo         : {RUTA_JSONL.stat().st_size // 1024} KB")

    # Estructura del primer ejemplo
    primero = ejemplos[0]
    print(f"\n{'─'*65}")
    print("📋 ESTRUCTURA DE CADA EJEMPLO")
    print("─" * 65)
    print(f"\n   Campos disponibles: {list(primero.keys())}")

    # ── Caso 1: tiene campo "messages" (formato conversacional) ──────────────
    if "messages" in primero:
        print(f"\n   Formato: CONVERSACIONAL (system / user / assistant)")
        print(f"   Roles encontrados en el primer ejemplo:")
        for msg in primero["messages"]:
            rol      = msg.get("role", "?")
            contenido = msg.get("content", "")
            print(f"\n   [{rol.upper():>10}]")
            print(f"   {contenido[:200]}{'...' if len(contenido) > 200 else ''}")

    # ── Caso 2: tiene campos directos question/answer ────────────────────────
    elif "question" in primero or "prompt" in primero:
        print(f"\n   Formato: PREGUNTA-RESPUESTA directa")
        for clave, valor in primero.items():
            if isinstance(valor, str):
                print(f"\n   [{clave.upper()}]")
                print(f"   {valor[:200]}{'...' if len(valor) > 200 else ''}")

    # ── Mostrar 5 ejemplos completos ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("📄 5 EJEMPLOS COMPLETOS")
    print("─" * 65)

    for i, ej in enumerate(ejemplos[:5]):
        print(f"\n  ┌─ Ejemplo #{i+1}")

        if "messages" in ej:
            for msg in ej["messages"]:
                rol      = msg.get("role", "?").upper()
                contenido = msg.get("content", "").strip()

                # Identificar si es pregunta o respuesta
                if rol == "USER":
                    print(f"  │  ❓ PREGUNTA  : {contenido[:150]}")
                elif rol == "ASSISTANT":
                    print(f"  │  ✅ RESPUESTA : {contenido[:150]}")
                elif rol == "SYSTEM":
                    print(f"  │  ⚙️  SISTEMA   : {contenido[:100]}")
        else:
            for clave, valor in ej.items():
                if isinstance(valor, str):
                    print(f"  │  {clave.upper():>12} : {valor[:150]}")

        print(f"  └{'─'*60}")

    # ── Estadísticas ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("📊 ESTADÍSTICAS")
    print("─" * 65)

    if "messages" in primero:
        longitudes_pregunta  = []
        longitudes_respuesta = []

        for ej in ejemplos:
            for msg in ej.get("messages", []):
                if msg.get("role") == "user":
                    longitudes_pregunta.append(len(msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    longitudes_respuesta.append(len(msg.get("content", "")))

        if longitudes_pregunta:
            print(f"\n   Preguntas:")
            print(f"   → Total                : {len(longitudes_pregunta)}")
            print(f"   → Longitud promedio    : {sum(longitudes_pregunta)//len(longitudes_pregunta)} chars")
            print(f"   → Más corta            : {min(longitudes_pregunta)} chars")
            print(f"   → Más larga            : {max(longitudes_pregunta)} chars")

        if longitudes_respuesta:
            print(f"\n   Respuestas:")
            print(f"   → Total                : {len(longitudes_respuesta)}")
            print(f"   → Longitud promedio    : {sum(longitudes_respuesta)//len(longitudes_respuesta)} chars")
            print(f"   → Más corta            : {min(longitudes_respuesta)} chars")
            print(f"   → Más larga            : {max(longitudes_respuesta)} chars")

    print(f"\n{'─'*65}")
    print("🔑 CÓMO SE USARÁ EN EL PROYECTO")
    print("─" * 65)
    print("""
   Este dataset tiene pares pregunta-respuesta sobre contratos.

   En el hackathon lo usarás para:
   → Demo en vivo: tomar las preguntas y hacérselas al RAG
   → Comparar: respuesta del RAG vs respuesta del dataset
   → Validación rápida sin esperar el procesamiento del CUAD

   Ventaja: las preguntas ya están formuladas en lenguaje natural,
   igual que lo haría un abogado o compliance officer real.
    """)

# ════════════════════════════════════════════════════════════════════════════
# 2. PDF WHO SOP
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("🏥 PARTE 2 — PDF WHO SOP (Procedimientos OMS)")
print("=" * 65)

RUTA_PDF = Path("data/who_sop/who_mch_clinic_sop.pdf")

if not RUTA_PDF.exists():
    print(f"\n❌ No se encontró: {RUTA_PDF}")
    print("   Asegúrate de haber ejecutado descargardatos.py")
else:
    print(f"\n✅ PDF encontrado: {RUTA_PDF}")
    print(f"   Tamaño: {RUTA_PDF.stat().st_size // 1024} KB")

    # Intentar leer con pdfplumber
    try:
        import pdfplumber

        print(f"\n{'─'*65}")
        print("📋 ANÁLISIS DEL PDF")
        print("─" * 65)

        with pdfplumber.open(RUTA_PDF) as pdf:
            total_paginas = len(pdf.pages)
            print(f"\n   Total de páginas : {total_paginas}")

            # Extraer texto de todas las páginas
            texto_completo = ""
            textos_por_pagina = []

            for pagina in pdf.pages:
                texto = pagina.extract_text() or ""
                textos_por_pagina.append(texto)
                texto_completo += texto + "\n"

            print(f"   Total caracteres : {len(texto_completo):,}")
            print(f"   Total palabras   : {len(texto_completo.split()):,}")

            # Mostrar primeras 3 páginas
            print(f"\n{'─'*65}")
            print("📄 CONTENIDO — PRIMERAS 3 PÁGINAS")
            print("─" * 65)

            for i, texto in enumerate(textos_por_pagina[:3]):
                print(f"\n  ┌─ PÁGINA {i+1}")
                lineas = [l.strip() for l in texto.split("\n") if l.strip()]
                for linea in lineas[:12]:
                    print(f"  │  {linea[:100]}")
                if len(lineas) > 12:
                    print(f"  │  ... ({len(lineas)-12} líneas más)")
                print(f"  └{'─'*60}")

            # Buscar secciones / títulos
            print(f"\n{'─'*65}")
            print("📑 SECCIONES DETECTADAS")
            print("─" * 65)

            secciones = []
            for linea in texto_completo.split("\n"):
                linea = linea.strip()
                # Detectar títulos: líneas cortas en mayúsculas o numeradas
                if (linea and len(linea) < 100 and
                    (linea.isupper() or
                     linea[:2].isdigit() or
                     linea.startswith("SOP") or
                     linea.startswith("Section") or
                     linea.startswith("SECTION") or
                     linea.startswith("Purpose") or
                     linea.startswith("Scope") or
                     linea.startswith("Procedure"))):
                    secciones.append(linea)

            secciones_unicas = list(dict.fromkeys(secciones))[:20]
            print(f"\n   Secciones/títulos detectados ({len(secciones_unicas)}):")
            for s in secciones_unicas:
                print(f"   → {s[:90]}")

            # Guardar texto extraído para usarlo en el RAG
            ruta_txt = Path("data/who_sop/who_sop_texto.txt")
            with open(ruta_txt, "w", encoding="utf-8") as f:
                f.write(texto_completo)

            print(f"\n   ✅ Texto extraído guardado en: {ruta_txt}")
            print(f"      ({len(texto_completo):,} caracteres listos para indexar en Azure AI Search)")

    except ImportError:
        print("\n   ⚠️  pdfplumber no instalado. Ejecuta:")
        print("      pip install pdfplumber")
        print("   Luego vuelve a correr este script.")

    except Exception as e:
        print(f"\n   ❌ Error leyendo el PDF: {e}")

    print(f"\n{'─'*65}")
    print("🔑 CÓMO SE USARÁ EN EL PROYECTO")
    print("─" * 65)
    print("""
   Este PDF es el factor multi-dominio del demo.

   En el hackathon lo usarás para:
   → Indexar en Azure AI Search junto con los contratos CUAD
   → Demostrar que el RAG responde sobre SALUD además de legal
   → Ejemplo de pregunta: "¿Cuál es el procedimiento para
     el triaje de pacientes según el SOP de la OMS?"
   → El jurado verá que el sistema funciona en 2 dominios
     sin cambiar una línea de código → INNOVACIÓN 🏆

   Para el Risk Scanner:
   → Detectar si el SOP tiene todas las secciones obligatorias
     (Purpose, Scope, Procedure, Responsibilities, References)
   → Generar score de completitud del protocolo
    """)

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("🎯 RESUMEN — Los 3 datasets listos para el RAG")
print("=" * 65)
print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Dataset              │ Uso principal               │
  ├─────────────────────────────────────────────────────┤
  │  CUAD (408 contratos) │ Base de conocimiento + RAGAS│
  │  Synthetic Q&A        │ Demo en vivo + validación   │
  │  WHO SOP (PDF)        │ Multi-dominio salud 🏆      │
  └─────────────────────────────────────────────────────┘

  Archivos listos para Azure AI Search (Día 2):
  → data/cuad/cuad_muestra_50.json    (50 contratos)
  → data/synthetic_qa/*.jsonl         (Q&A sintético)
  → data/who_sop/who_sop_texto.txt    (SOP en texto)

  Próximo paso:
  git add .
  git commit -m "Dia 1 completo: todos los datasets explorados"
  git push origin main
""")
print("➡️  DÍA 1 100% COMPLETADO — Siguiente: Día 2 Azure AI Search")