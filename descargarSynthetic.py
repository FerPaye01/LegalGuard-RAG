"""
Descarga del dataset Synthetic Q&A directamente desde HuggingFace
"""

import requests
from pathlib import Path
from tqdm import tqdm

DIR_SYNTH = Path("data/synthetic_qa")
DIR_SYNTH.mkdir(parents=True, exist_ok=True)

# URL correcta con los caracteres especiales manejados
url  = "https://huggingface.co/datasets/strova-ai/legal_contract_dataset/resolve/main/legal-contract%20(2).jsonl"
ruta = DIR_SYNTH / "synthetic_qa_train.jsonl"

print("=" * 60)
print("💬 Descargando Synthetic Q&A (strova-ai)...")
print("=" * 60)
print(f"\n📥 URL: {url}\n")

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(url, stream=True, timeout=120, headers=headers)
r.raise_for_status()

tamano = int(r.headers.get("content-length", 0))

with open(ruta, "wb") as f, tqdm(
    desc="   Progreso",
    total=tamano,
    unit="B", unit_scale=True, unit_divisor=1024
) as barra:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
        barra.update(len(chunk))

# ── Verificar el contenido descargado ───────────────────────────────────────
import json

with open(ruta, "r", encoding="utf-8") as f:
    lineas = [l.strip() for l in f if l.strip()]

print(f"\n✅ Archivo descargado: {ruta}")
print(f"   Tamaño  : {ruta.stat().st_size // 1024} KB")
print(f"   Ejemplos: {len(lineas)} registros")

# Mostrar los primeros 3 ejemplos
print(f"\n{'─'*60}")
print("📋 PRIMEROS 3 EJEMPLOS")
print("─" * 60)

for i, linea in enumerate(lineas[:3]):
    try:
        ejemplo = json.loads(linea)
        print(f"\n  Ejemplo #{i+1}")
        print(f"  Campos: {list(ejemplo.keys())}")

        # Si tiene campo 'messages' (formato conversacional)
        if "messages" in ejemplo:
            for msg in ejemplo["messages"]:
                rol      = msg.get("role", "?")
                contenido = msg.get("content", "")[:120]
                print(f"  [{rol:>9}]: {contenido}...")

        # Si tiene otros campos
        else:
            for clave, valor in ejemplo.items():
                if isinstance(valor, str):
                    print(f"  {clave}: {valor[:120]}")

    except json.JSONDecodeError:
        print(f"  ⚠️ Línea {i+1} no es JSON válido")

print(f"\n{'='*60}")
print("🎉 ¡Synthetic Q&A descargado correctamente!")
print(f"{'='*60}")
print(f"""
  ✅ Guardado en: {ruta}
  ✅ {len(lineas)} ejemplos listos para el demo

  🔑 Para tu RAG:
     Este dataset tiene preguntas y respuestas sobre contratos
     en formato conversacional — perfecto para el demo en vivo.
""")
print("➡️  Siguiente: python cuad_directo.py")