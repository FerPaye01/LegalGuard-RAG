# 🚀 Guía de Despliegue en Azure App Service

LegalGuard RAG está diseñado para ejecutarse en Azure. Sigue estos pasos para llevar tu entorno local a la nube.

## 1. Preparación del Recurso
1. Crea un **App Service** en Azure (Plan B1 o superior recomendado).
2. Recomendamos **Linux** con Python 3.10/3.11.
3. Habilita **Application Insights** durante la creación para obtener telemetría.

## 2. Variables de Entorno (App Settings)
En el portal de Azure, ve a **Configuración > Configuración de la aplicación** y añade todas las variables de tu `.env`:
- `AZURE_OPENAI_API_KEY`, `AZURE_SEARCH_ENDPOINT`, etc.
- **IMPORTANTE**: Añade `AZURE_MONITOR_CONNECTION_STRING` para que los logs aparezcan en Application Insights.

## 3. Comando de Inicio (Linux)
Si usas Linux, configura este comando de inicio:
```bash
python -m streamlit run src/frontend/streamlit_app.py --server.port 8000 --server.address 0.0.0.0
```

## 4. Configuración Windows (web.config)
Si despliegas en Windows, asegúrate de incluir el archivo `deployment/web.config` en la raíz de tu sitio.

## 5. CI/CD con GitHub Actions
Puedes usar el asistente de despliegue de Azure para generar un workflow que despliegue automáticamente cada que hagas `push` a `main`.

---
**Nota sobre Puertos**: Azure App Service mapea internamente el tráfico al puerto 80/443, pero tu app debe escuchar en el puerto que Azure te asigne (generalmente 8000 o el detectado por Streamlit).
