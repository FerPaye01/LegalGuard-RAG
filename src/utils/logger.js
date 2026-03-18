/**
 * Sistema de Logging del Proyecto
 * Proporciona diferentes niveles de logs para seguimiento técnico y comprensible
 */

// Configuración de colores para consola
const COLORS = {
    debug: 'color: #6B7280; font-style: italic',
    info: 'color: #3B82F6; font-weight: bold',
    sequence: 'color: #10B981; font-weight: bold',
    warn: 'color: #F59E0B; font-weight: bold',
    error: 'color: #EF4444; font-weight: bold'
}

// Contador de secuencia para seguir el flujo de ejecución
let sequenceCounter = 0

/**
 * Log debug - Detalles técnicos para desarrolladores
 */
function logDebug(message, data = null) {
    const timestamp = new Date().toLocaleTimeString()
    if (data) {
        console.log(`%c[DEBUG ${timestamp}] ${message}`, COLORS.debug, data)
    } else {
        console.log(`%c[DEBUG ${timestamp}] ${message}`, COLORS.debug)
    }
}

/**
 * Log info - Información general del sistema
 */
function logInfo(message) {
    const timestamp = new Date().toLocaleTimeString()
    console.log(`%c[INFO ${timestamp}] ${message}`, COLORS.info)
}

/**
 * Log sequence - Muestra el flujo de ejecución de forma comprensible
 */
function logSequence(action, detail = '') {
    sequenceCounter++
    const timestamp = new Date().toLocaleTimeString()
    const detailMessage = detail ? ` → ${detail}` : ''
    console.log(`%c[STEP ${sequenceCounter}] ${timestamp} | ${action}${detailMessage}`, COLORS.sequence)
}

/**
 * Log warn - Situaciones inesperadas pero manejables
 */
function logWarn(message, context = '') {
    const timestamp = new Date().toLocaleTimeString()
    const ctx = context ? ` (${context})` : ''
    console.warn(`%c[WARN ${timestamp}] ${message}${ctx}`, COLORS.warn)
}

/**
 * Log error - Errores críticos del sistema
 */
function logError(message, error = null) {
    const timestamp = new Date().toLocaleTimeString()
    console.error(`%c[ERROR ${timestamp}] ${message}`, COLORS.error)
    if (error) {
        console.error(error)
    }
}

/**
 * Reinicia el contador de secuencia (útil al iniciar nueva operación)
 */
function resetSequence() {
    sequenceCounter = 0
    logInfo('Secuencia reiniciada')
}

/**
 * Muestra resumen de la secuencia ejecutada
 */
function sequenceSummary(operation) {
    console.log(`%c═══════════════════════════════════════`, 'color: #8B5CF6')
    console.log(`%c RESUMEN: ${operation}`, 'color: #8B5CF6; font-weight: bold; font-size: 14px')
    console.log(`%c Total de pasos ejecutados: ${sequenceCounter}`, 'color: #8B5CF6')
    console.log(`%c═══════════════════════════════════════`, 'color: #8B5CF6')
}

// Exportar funciones
export {
    logDebug,
    logInfo,
    logSequence,
    logWarn,
    logError,
    resetSequence,
    sequenceSummary
}
