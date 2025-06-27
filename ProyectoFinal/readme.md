## SW de simulación
Webots

## Descripción del Proyecto

El presente proyecto tiene como objetivo el desarrollo e implementación de un robot móvil autónomo en el simulador Webots, utilizando control cinemático diferencial y sensores para la percepción del entorno, por último mencionar que el entorno que se hizo el trabajo hecho el robot móvil fue en lenguaje Python.

**El robot es capaz de:**

    - Detectar y evitar obstáculos en tiempo real mediante la combinación de un sensor LIDAR 2D y sensores de distancia (ultrasónicos o infrarrojos).

    - Construir un mapa local del entorno (grilla de ocupación 8x8) a partir de los datos del LIDAR.

    - Planificar rutas óptimas en el mapa utilizando el algoritmo de planificación A* (A-Star).

    - Navegar de manera autónoma hacia un objetivo definido en el entorno simulado, ajustando su trayectoria en función de la percepción actualizada.

El desarrollo integra percepción, planificación y control en un entorno dinámico, permitiendo que el robot reaccione ante cambios en su entorno y tome decisiones en tiempo real. El robot se prueba en un entorno controlado de 4m x 4m con obstáculos distribuidos, evaluando su desempeño en términos de eficiencia de navegación, precisión de planificación y robustez en la evasión de obstáculos.

## Arquitectura del Software

El sistema de control del robot se organiza en tres niveles principales:

### 1️⃣ Percepción

- **LIDAR 2D (128 resoluciones):**
  - Obtiene un mapa parcial del entorno en cada ciclo.
  - Detecta obstáculos en un rango de hasta 1 metro.
  - Construye un mapa de ocupación (grilla 8x8).

- **Sensores de distancia (ultrasónicos/infrarrojos):**
  - Detectan obstáculos cercanos (frontales).
  - Proveen una capa de seguridad adicional para la evasión reactiva.

- **GPS:**
  - Obtiene la posición actual del robot en el entorno simulado (coordenadas X, Z).

### 2️⃣ Planificación

- **Grilla de ocupación:**
  - Mapa 2D representado como una matriz `grid[8][8]`.
  - Celdas marcadas como libres u ocupadas.

- **Algoritmo de planificación A\* (A-Star):**
  - Calcula la ruta óptima desde la posición actual hacia el objetivo.
  - Se ejecuta en cada ciclo para ajustar el plan en función de los cambios en el mapa.

### 3️⃣ Control

- **Control de navegación:**
  - Si se detecta un obstáculo cercano → evasión reactiva (giro en el lugar).
  - Si no hay obstáculos cercanos → seguimiento del camino planificado (A\*).

- **Controlador de motores:**
  - Control cinemático diferencial (velocidad de ruedas izquierda/derecha).
  - Ajustes simples de velocidad en función del ángulo hacia el waypoint actual.

Esta arquitectura modular permite la integración de múltiples fuentes de percepción para una navegación robusta, combinando planificación deliberativa (A\*) con comportamientos reactivos (evasión rápida).
## Resultados

### Métricas de desempeño

- **Tiempo total de navegación:** ___ segundos
- **Longitud del path (celdas):** ___
- **Tiempo de planificación (A\*):** ___ milisegundos
- **Porcentaje del mapa explorado:** ___ %

### Análisis de algoritmos

- **Precisión:** El algoritmo A\* logra rutas óptimas en escenarios conocidos.
- **Eficiencia:** La planificación en grilla 8x8 es rápida (< 50 ms por iteración).
- **Robustez:** La combinación LIDAR + Distance Sensor permite evitar obstáculos dinámicos.

### Reflexión sobre mejoras

- Implementar control proporcional hacia el waypoint.
- Usar SLAM o un mapeo más denso para escenarios más complejos.
- Integrar planificación incremental para ambientes dinámicos.

