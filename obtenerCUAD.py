"""
LegalGuard RAG — CUAD completo: descarga, extracción y procesamiento
Ejecutar: python cuad_completo.py

Hace todo en orden:
  1. Descarga data.zip desde GitHub oficial
  2. Extrae los 3 JSON (CUADv1, train_separate_questions, test)
  3. Procesa y convierte al formato RAG
  4. Guarda cuad_plano.json y clausulas_tipos.json
  5. Muestra 5 ejemplos reales con cláusulas
"""

import json
import zipfile
import requests
import io
from pathlib import Path
from tqdm import tqdm

# ── Configuración ────────────────────────────────────────────────────────────
DIR_CUAD = Path("data/cuad")
DIR_CUAD.mkdir(parents=True, exist_ok=True)

URL_ZIP  = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
HEADERS  = {"User-Agent": "Mozilla/5.0"}

# ════════════════════════════════════════════════════════════════════════════
# PASO 1 — DESCARGAR data.zip
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("📦 LegalGuard RAG — CUAD completo")
print("=" * 65)
print(f"\n[1/4] Descargando data.zip desde GitHub oficial...")
print(f"      {URL_ZIP}\n")

ruta_zip = DIR_CUAD / "data.zip"

try:
    r = requests.get(URL_ZIP, stream=True, timeout=300, headers=HEADERS)
    r.raise_for_status()
    tamano = int(r.headers.get("content-length", 0))

    with open(ruta_zip, "wb") as f, tqdm(
        desc="   Descargando",
        total=tamano,
        unit="B", unit_scale=True, unit_divisor=1024
    ) as barra:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            barra.update(len(chunk))

    print(f"\n   ✅ data.zip ({ruta_zip.stat().st_size // 1024:,} KB)")

except Exception as e:
    print(f"\n   ❌ Error descargando: {e}")
    print("   Asegúrate de tener conexión a internet activa.")
    exit(1)

# ════════════════════════════════════════════════════════════════════════════
# PASO 2 — EXTRAER EL ZIP
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[2/4] Extrayendo archivos del ZIP...")

with zipfile.ZipFile(ruta_zip, "r") as zf:
    archivos_zip = zf.namelist()
    print(f"\n   Archivos encontrados dentro del ZIP:")
    for nombre in archivos_zip:
        print(f"   → {nombre}")

    # Extraer todos en data/cuad/
    zf.extractall(DIR_CUAD)

print(f"\n   ✅ Extraído en {DIR_CUAD}/")

# Eliminar el ZIP (ya no lo necesitamos)
ruta_zip.unlink()
print(f"   🗑️  data.zip eliminado")

# ════════════════════════════════════════════════════════════════════════════
# PASO 3 — LOCALIZAR LOS JSON EXTRAÍDOS
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[3/4] Localizando archivos JSON extraídos...")

# Buscar todos los JSON en data/cuad/ incluyendo subcarpetas
todos_json = list(DIR_CUAD.rglob("*.json"))

print(f"\n   JSONs encontrados:")
for j in todos_json:
    print(f"   → {j.relative_to(DIR_CUAD)}  ({j.stat().st_size // 1024:,} KB)")

# Identificar cuál es el de train (el más grande generalmente)
archivo_train = None
archivo_test  = None
archivo_cuadv1 = None

for j in todos_json:
    nombre = j.name.lower()
    if "train_separate" in nombre:
        archivo_train  = j
    elif "cuadv1" in nombre or "cuad_v1" in nombre:
        archivo_cuadv1 = j
    elif "test" in nombre:
        archivo_test   = j

# Si no encontró train_separate, usar CUADv1 como alternativa
archivo_principal = archivo_train or archivo_cuadv1 or todos_json[0]
print(f"\n   📖 Archivo principal a procesar: {archivo_principal.name}")

# ════════════════════════════════════════════════════════════════════════════
# PASO 4 — PROCESAR Y CONVERTIR AL FORMATO RAG
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[4/4] Procesando contratos para el RAG...")
print("─" * 65)

with open(archivo_principal, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\n   Estructura del JSON: {list(data.keys())}")
total_docs = len(data.get("data", []))
print(f"   Total de documentos: {total_docs}")

# Procesar cada contrato
contratos_planos = []
preguntas_unicas = set()

print(f"\n   Procesando...")
for documento in tqdm(data.get("data", []), desc="   Contratos"):
    titulo = documento.get("title", "Sin título")

    for parrafo in documento.get("paragraphs", []):
        contexto            = parrafo.get("context", "")
        clausulas_presentes = []
        clausulas_ausentes  = []

        for qa in parrafo.get("qas", []):
            pregunta   = qa.get("question", "")
            respuestas = qa.get("answers", [])
            preguntas_unicas.add(pregunta)

            if respuestas and respuestas[0].get("text", "").strip():
                clausulas_presentes.append({
                    "clausula" : pregunta,
                    "texto"    : respuestas[0]["text"],
                    "inicio"   : respuestas[0]["answer_start"]
                })
            else:
                clausulas_ausentes.append(pregunta)

        total_cl = len(clausulas_presentes) + len(clausulas_ausentes)
        score    = round(len(clausulas_ausentes) / max(total_cl, 1) * 100)

        contratos_planos.append({
            "titulo"              : titulo,
            "context"             : contexto,
            "clausulas_presentes" : clausulas_presentes,
            "clausulas_ausentes"  : clausulas_ausentes,
            "total_clausulas"     : total_cl,
            "score_riesgo"        : score
        })

# ── Guardar formato plano completo
ruta_plano = DIR_CUAD / "cuad_plano.json"
with open(ruta_plano, "w", encoding="utf-8") as f:
    json.dump(contratos_planos, f, ensure_ascii=False, indent=2)

# ── Guardar muestra de 50 para el hackathon (más rápido de indexar)
ruta_muestra = DIR_CUAD / "cuad_muestra_50.json"
with open(ruta_muestra, "w", encoding="utf-8") as f:
    json.dump(contratos_planos[:50], f, ensure_ascii=False, indent=2)

# ── Guardar lista de cláusulas
ruta_clausulas = DIR_CUAD / "clausulas_tipos.json"
with open(ruta_clausulas, "w", encoding="utf-8") as f:
    json.dump(sorted(list(preguntas_unicas)), f, ensure_ascii=False, indent=2)

print(f"\n   ✅ cuad_plano.json         ({len(contratos_planos)} contratos)")
print(f"   ✅ cuad_muestra_50.json    (50 contratos para el hackathon)")
print(f"   ✅ clausulas_tipos.json    ({len(preguntas_unicas)} tipos de cláusulas)")

# ════════════════════════════════════════════════════════════════════════════
# MOSTRAR 5 EJEMPLOS REALES
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("📄 5 CONTRATOS DE EJEMPLO REALES")
print("─" * 65)

for i, c in enumerate(contratos_planos[:5]):
    nivel = ("🔴 ALTO"  if c['score_riesgo'] > 60 else
             "🟡 MEDIO" if c['score_riesgo'] > 30 else
             "🟢 BAJO")

    print(f"\n  ┌─ #{i+1} {c['titulo'][:60]}")
    print(f"  │  📄 Texto    : {len(c['context']):,} caracteres")
    print(f"  │  ✅ Presentes: {len(c['clausulas_presentes'])} cláusulas")
    print(f"  │  ❌ Ausentes : {len(c['clausulas_ausentes'])} cláusulas")
    print(f"  │  📊 Riesgo   : {c['score_riesgo']}/100  {nivel}")

    if c["clausulas_presentes"]:
        ej = c["clausulas_presentes"][0]
        print(f"  │")
        print(f"  │  Ejemplo cláusula encontrada:")
        print(f"  │  → Tipo : {ej['clausula'][:65]}")
        print(f"  └  → Texto: \"{ej['texto'][:130]}\"")
    else:
        print(f"  └  (sin cláusulas identificadas en este fragmento)")

# ════════════════════════════════════════════════════════════════════════════
# LAS CLÁUSULAS DETECTADAS
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"⚖️  LAS {len(preguntas_unicas)} CLÁUSULAS DEL CUAD")
print("─" * 65)
print("(Estas son las que detectará tu Risk Scanner)\n")

for i, q in enumerate(sorted(preguntas_unicas), 1):
    print(f"  {i:>3}. {q}")

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("🎉 ¡CUAD COMPLETAMENTE LISTO!")
print(f"{'='*65}")

print(f"""
  ✅ {len(contratos_planos)} contratos procesados
  ✅ {len(preguntas_unicas)} tipos de cláusulas detectadas
  ✅ Archivos guardados en data/cuad/:

     📄 cuad_plano.json         ← todos los contratos en formato RAG
     📄 cuad_muestra_50.json    ← muestra para Azure AI Search
     📄 clausulas_tipos.json    ← las {len(preguntas_unicas)} cláusulas del Risk Scanner
     📄 CUADv1.json             ← raw original
     📄 train_separate_questions.json ← raw original
     📄 test.json               ← raw original

  🔑 Cómo se usará cada archivo:
     cuad_muestra_50.json    → indexar en Azure AI Search (Día 2)
     clausulas_tipos.json    → Risk Scanner detecta estas cláusulas
     cuad_plano.json         → ground truth para métricas RAGAS

  Ahora sube a GitHub:
  ─────────────────────────────────────
  git add .
  git commit -m "Dia 1 completo: CUAD descargado y procesado"
  git push origin main
  ─────────────────────────────────────
""")
print("➡️  DÍA 1 COMPLETADO — Siguiente: Día 2 configurar Azure AI Search")