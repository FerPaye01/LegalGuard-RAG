"""
LegalGuard RAG — Día 1: Descarga de datasets
Ejecutar: descargardatos.py

Descarga:
  1. CUAD (Atticus Project) desde HuggingFace
  2. Synthetic Q&A (strova-ai) desde HuggingFace
  3. PDF SOP de la OMS

Requisito previo:
  pip install datasets requests tqdm
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ── Directorios destino ─────────────────────────────────────────────────────
DIR_CUAD     = Path("data/cuad")
DIR_SYNTH    = Path("data/synthetic_qa")
DIR_WHO      = Path("data/who_sop")

for d in [DIR_CUAD, DIR_SYNTH, DIR_WHO]:
    d.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. DESCARGAR CUAD
# ════════════════════════════════════════════════════════════════════════════
def descargar_cuad():
    print("\n" + "═" * 60)
    print("📄 [1/3] Descargando CUAD (Atticus Project)...")
    print("═" * 60)

    from datasets import load_dataset

    ds = load_dataset(
        "theatticusproject/cuad",
        trust_remote_code=True,
        verification_mode="no_checks"
    )

    print(f"\n✅ Dataset cargado:")
    print(f"   Splits disponibles : {list(ds.keys())}")
    print(f"   Split 'train'      : {len(ds['train'])} ejemplos")
    print(f"   Columnas           : {ds['train'].column_names}")

    # ── Guardar solo train (no existe split test en este dataset) ──
    ruta_train = DIR_CUAD / "cuad_train.json"
    print(f"\n💾 Guardando en {DIR_CUAD}/ ...")
    ds["train"].to_json(str(ruta_train))

    # Guardar muestra de 50 para trabajar más rápido
    muestra = ds["train"].select(range(min(50, len(ds["train"]))))
    muestra.to_json(str(DIR_CUAD / "cuad_muestra_50.json"))

    print(f"   ✅ cuad_train.json")
    print(f"   ✅ cuad_muestra_50.json  (50 contratos para el hackathon)")

    return ds

# ════════════════════════════════════════════════════════════════════════════
# 2. DESCARGAR SYNTHETIC Q&A
# ════════════════════════════════════════════════════════════════════════════
def descargar_synthetic_qa():
    print("\n" + "═" * 60)
    print("💬 [2/3] Descargando Synthetic Q&A (strova-ai)...")
    print("═" * 60)

    import requests
    from tqdm import tqdm

    # Descarga directa del archivo JSONL desde HuggingFace
    url = "https://huggingface.co/datasets/strova-ai/legal_contract_dataset/resolve/main/legal-contract%20(2).jsonl"

    ruta = DIR_SYNTH / "synthetic_qa_train.jsonl"

    print(f"\n📥 Descargando archivo JSONL directamente...")

    try:
        respuesta = requests.get(url, stream=True, timeout=60)
        respuesta.raise_for_status()

        tamano = int(respuesta.headers.get("content-length", 0))

        with open(ruta, "wb") as f, tqdm(
            desc="   Progreso",
            total=tamano,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as barra:
            for chunk in respuesta.iter_content(chunk_size=8192):
                f.write(chunk)
                barra.update(len(chunk))

        # Verificar cuántos ejemplos se descargaron
        with open(ruta, "r", encoding="utf-8") as f:
            lineas = [l for l in f if l.strip()]

        print(f"\n✅ Descarga exitosa:")
        print(f"   📄 synthetic_qa_train.jsonl")
        print(f"   📊 {len(lineas)} ejemplos descargados")

        # Mostrar el primer ejemplo para verificar estructura
        import json
        primer = json.loads(lineas[0])
        print(f"\n   Vista previa del primer ejemplo:")
        print(f"   Campos: {list(primer.keys())}")

        # Si tiene campo 'messages', mostrar roles
        if "messages" in primer:
            for msg in primer["messages"]:
                rol = msg.get("role", "?")
                contenido = msg.get("content", "")[:80]
                print(f"   [{rol}]: {contenido}...")

    except requests.exceptions.RequestException as e:
        print(f"\n   ⚠️  Error en descarga directa: {e}")
        print("\n   👉 Descarga manual:")
        print("      1. Ve a: https://huggingface.co/datasets/strova-ai/legal_contract_dataset")
        print("      2. Clic en 'Files and versions'")
        print("      3. Descarga el archivo .jsonl")
        print(f"      4. Guárdalo en: {ruta}")

    return ruta

# ════════════════════════════════════════════════════════════════════════════
# 3. DESCARGAR PDF SOP OMS
# ════════════════════════════════════════════════════════════════════════════
def descargar_who_sop():
    print("\n" + "═" * 60)
    print("🏥 [3/3] Descargando PDF SOP de la OMS...")
    print("   Procedimientos Operativos Estándar · Dominio Salud")
    print("═" * 60)

    url = (
        "https://platform.who.int/docs/default-source/mca-documents/"
        "policy-documents/operational-guidance/"
        "MHL-MN-48-02-OPERATIONALGUIDANCE-eng-MCH-Clinic-SOPs.pdf"
    )

    ruta_pdf = DIR_WHO / "who_mch_clinic_sop.pdf"

    print(f"\n📥 Descargando desde:\n   {url}\n")

    try:
        respuesta = requests.get(url, stream=True, timeout=60)
        respuesta.raise_for_status()

        tamano_total = int(respuesta.headers.get("content-length", 0))

        with open(ruta_pdf, "wb") as f, tqdm(
            desc="   Progreso",
            total=tamano_total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as barra:
            for chunk in respuesta.iter_content(chunk_size=8192):
                f.write(chunk)
                barra.update(len(chunk))

        print(f"\n   ✅ who_mch_clinic_sop.pdf ({ruta_pdf.stat().st_size // 1024} KB)")

    except requests.exceptions.RequestException as e:
        print(f"\n   ⚠️  No se pudo descargar automáticamente: {e}")
        print("   👉 Descárgalo manualmente desde:")
        print(f"      {url}")
        print(f"   👉 Guárdalo en: {ruta_pdf}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 LegalGuard RAG — Descarga de datasets (Día 1)")
    print("=" * 60)

    ds_cuad  = descargar_cuad()
    ds_synth = descargar_synthetic_qa()
    descargar_who_sop()

    print("\n" + "=" * 60)
    print("🎉 ¡Todos los datasets descargados!")
    print("=" * 60)
    print("\nResumen de archivos creados:")
    for archivo in sorted(Path("data").rglob("*")):
        if archivo.is_file():
            kb = archivo.stat().st_size // 1024
            print(f"   📄 {archivo}  ({kb} KB)")

    print("\n➡️  Próximo paso: python dia1_explorar_cuad.py")