import math
import numpy as np
from controller import Robot
from enum import Enum
import heapq

# --- CONSTANTES ---
# Estas son las variables de configuración principales del robot.
# Ajustar estos valores ("tuning") es clave para perfeccionar el comportamiento.
TIME_STEP = 64
GRID_SIZE = 20
CELL_SIZE = 0.25
SPEED = 4.0
EVASION_TURN_SPEED = 10.0
GOAL_THRESHOLD = 0.2
DS_THRESHOLD = 970.0 
ESCAPE_STEPS = 25

# --- ESTADOS DEL ROBOT ---
# Usamos una máquina de estados para controlar el comportamiento general del robot.
# Esto evita que los diferentes comportamientos (navegar, evadir) entren en conflicto.
class RobotState(Enum):
    NAVIGATING = 1
    AVOIDING_OBSTACLE = 2
    GOAL_REACHED = 3

# --- CLASES AUXILIARES ---
# Estas clases nos ayudan a organizar los datos de una manera más limpia y legible.
class Point:
    """Representa un punto en la grilla (coordenadas enteras)."""
    def __init__(self, x=0, y=0): self.x, self.y = x, y
    def __eq__(self, other): return self.x == other.x and self.y == other.y
    def __hash__(self): return hash((self.x, self.y))

class Node:
    """Representa un nodo para el algoritmo A*, conteniendo costos y el nodo padre."""
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x, self.y, self.g, self.h, self.parent = x, y, g, h, parent
        self.f = g + h # Costo total F = G (costo desde el inicio) + H (heurística al objetivo)
    def __lt__(self, other): return self.f < other.f
    def __eq__(self, other): return self.x == other.x and self.y == other.y

class Position:
    """Representa una posición en el mundo de Webots (coordenadas flotantes)."""
    def __init__(self, x=0.0, y=0.0): self.x, self.y = x, y

# --- CLASE PRINCIPAL DEL ROBOT ---
class RobotNavigator:
    """Clase principal que encapsula toda la lógica y el estado del robot."""
    def __init__(self):
        self.robot = Robot()
        self.robot_state = RobotState.NAVIGATING
        self.goal_pos = Position(1.0, 1.0) 
        
        # Variables de navegación y planificación
        self.path = []
        self.current_path_index = 0
        
        # Variables de estado y métricas
        self.prev_robot_pos = Position(0, 0)
        self.evasion_counter = 0 
        self.total_distance = 0.0 # Acumulador para la distancia recorrida
        self.total_planning_time = 0.0
        self.planning_runs = 0
        self.last_path_length = 0
        self.explored_grid = np.full((GRID_SIZE, GRID_SIZE), False, dtype=bool)

        self._setup_devices()
        print(f"Controlador iniciado. Objetivo: ({self.goal_pos.x:.2f}, {self.goal_pos.y:.2f})")

    def _setup_devices(self):
        """Inicializa y habilita todos los dispositivos del robot."""
        # Motores
        self.wheels = [self.robot.getDevice(name) for name in ["wheel1", "wheel2", "wheel3", "wheel4"]]
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
        
        # Sensores de distancia para evasión reactiva
        self.ds_left = self.robot.getDevice("ds_left"); self.ds_left.enable(TIME_STEP)
        self.ds_right = self.robot.getDevice("ds_right"); self.ds_right.enable(TIME_STEP)
        
        # Sensores para localización y mapeo
        self.lidar = self.robot.getDevice("lidar"); self.lidar.enable(TIME_STEP)
        self.gps = self.robot.getDevice("gps"); self.gps.enable(TIME_STEP)

    def world_to_grid(self, x, y):
        """Convierte coordenadas del mundo (flotantes) a celdas de la grilla (enteros)."""
        grid_x = int((x + GRID_SIZE * CELL_SIZE / 2) / CELL_SIZE)
        grid_y = int((y + GRID_SIZE * CELL_SIZE / 2) / CELL_SIZE)
        grid_x = max(0, min(grid_x, GRID_SIZE - 1)); grid_y = max(0, min(grid_y, GRID_SIZE - 1))
        return Point(grid_x, grid_y)

    def grid_to_world(self, x, y):
        """Convierte coordenadas de la grilla al centro de la celda en el mundo."""
        world_x = x * CELL_SIZE - (GRID_SIZE * CELL_SIZE / 2) + (CELL_SIZE / 2)
        world_y = y * CELL_SIZE - (GRID_SIZE * CELL_SIZE / 2) + (CELL_SIZE / 2)
        return Position(world_x, world_y)

    def heuristic(self, p1, p2):
        """Calcula la heurística para A* (distancia Manhattan)."""
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def plan_path_astar(self, grid, start, goal):
        """Implementación del algoritmo de planificación de rutas A*."""
        open_list, closed_set = [], set()
        start_node = Node(start.x, start.y, 0, self.heuristic(start, goal))
        heapq.heappush(open_list, start_node)
        
        while open_list:
            current = heapq.heappop(open_list)
            if (current.x, current.y) in closed_set: continue
            closed_set.add((current.x, current.y))
            
            if current.x == goal.x and current.y == goal.y:
                path = []
                while current: path.append(Point(current.x, current.y)); current = current.parent
                return path[::-1] # Devuelve el camino desde el inicio hasta el final
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]: # 4 direcciones
                nx, ny = current.x + dx, current.y + dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or grid[nx, ny] == 1 or (nx, ny) in closed_set: continue
                
                neighbor = Node(nx, ny, current.g + 1, self.heuristic(Point(nx,ny), goal), current)
                heapq.heappush(open_list, neighbor)
        
        return [] # Devuelve una lista vacía si no se encuentra ruta

    def set_speeds(self, left, right):
        """Establece la velocidad de las ruedas, limitándola al máximo permitido."""
        left = max(-10.0, min(left, 10.0)); right = max(-10.0, min(right, 10.0))
        self.wheels[0].setVelocity(left); self.wheels[1].setVelocity(right)
        self.wheels[2].setVelocity(left); self.wheels[3].setVelocity(right)

    def follow_path(self, robot_pos, robot_orientation):
        """Lógica para seguir la secuencia de puntos de la ruta calculada por A*."""
        if not self.path or self.current_path_index >= len(self.path):
            self.path = [] 
            return
        
        target_grid = self.path[self.current_path_index]
        target_world = self.grid_to_world(target_grid.x, target_grid.y)
        
        dist_to_target = math.sqrt((target_world.x - robot_pos.x)**2 + (target_world.y - robot_pos.y)**2)
        
        # Si estamos lo suficientemente cerca del waypoint actual, avanzamos al siguiente.
        if dist_to_target < GOAL_THRESHOLD:
            self.current_path_index += 1
            if self.current_path_index >= len(self.path): self.path = []; return
        
        # Control Proporcional: ajusta la velocidad de giro en proporción al error de ángulo.
        target_grid = self.path[self.current_path_index]
        target_world = self.grid_to_world(target_grid.x, target_grid.y)
        target_angle = math.atan2(target_world.y - robot_pos.y, target_world.x - robot_pos.x)
        angle_diff = target_angle - robot_orientation
        
        # Normaliza el ángulo para tomar siempre el giro más corto
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        turn_speed = angle_diff * 2.0
        self.set_speeds(SPEED - turn_speed, SPEED + turn_speed)

    def run(self):
        """El bucle principal del controlador que se ejecuta en cada paso de la simulación."""
        robot_orientation = 0.0
        planning_counter = 0
        debug_counter = 0
        start_time = self.robot.getTime()

        while self.robot.step(TIME_STEP) != -1:
            # --- 1. PERCEPCIÓN: Obtener información de los sensores ---
            gps_values = self.gps.getValues()
            robot_pos = Position(gps_values[0], gps_values[1])
            
            # Calcular la distancia recorrida en este paso y sumarla al total
            dx = robot_pos.x - self.prev_robot_pos.x
            dy = robot_pos.y - self.prev_robot_pos.y
            self.total_distance += math.sqrt(dx**2 + dy**2)
            
            # Deducir la orientación del robot a partir de su movimiento
            if math.sqrt(dx**2 + dy**2) > 0.001:
                robot_orientation = math.atan2(dy, dx)
            
            # Leer los sensores de distancia para la evasión reactiva
            left_obstacle = self.ds_left.getValue() < DS_THRESHOLD
            right_obstacle = self.ds_right.getValue() < DS_THRESHOLD

            # --- 2. CONTROL: Decidir qué hacer según el estado actual ---
            if self.robot_state == RobotState.NAVIGATING:
                if left_obstacle or right_obstacle:
                    # Si hay un obstáculo, cambia al estado de evasión.
                    self.robot_state = RobotState.AVOIDING_OBSTACLE
                    self.path = [] # La ruta anterior ya no es válida.
                    self.evasion_counter = ESCAPE_STEPS # Inicia el temporizador de la maniobra.
                    print("-> Obstáculo. Iniciando maniobra de evasión.")
                else:
                    # Si no hay peligro, navega hacia el objetivo.
                    planning_counter += 1
                    if planning_counter > 20 or not self.path:
                        # Replanifica la ruta periódicamente o si no tiene una.
                        print("Planificando ruta...")
                        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
                        ranges = self.lidar.getRangeImage()
                        resolution = self.lidar.getHorizontalResolution(); fov = self.lidar.getFov()
                        for i in range(resolution):
                            angle = robot_orientation - fov / 2.0 + i * (fov / resolution)
                            dist = ranges[i]
                            if math.isinf(dist) or math.isnan(dist) or dist > 3.0: continue
                            obs_x = robot_pos.x + dist * math.cos(angle); obs_y = robot_pos.y + dist * math.sin(angle)
                            obs_cell = self.world_to_grid(obs_x, obs_y); grid[obs_cell.x, obs_cell.y] = 1
                            self.explored_grid[obs_cell.x, obs_cell.y] = True

                        start = self.world_to_grid(robot_pos.x, robot_pos.y)
                        goal = self.world_to_grid(self.goal_pos.x, self.goal_pos.y)
                        grid[start.x, start.y] = 0 # Asegura que la celda de inicio esté libre.

                        # Medir tiempo de planificación
                        planning_start_time = self.robot.getTime()
                        self.path = self.plan_path_astar(grid, start, goal)
                        planning_end_time = self.robot.getTime()
                        self.total_planning_time += (planning_end_time - planning_start_time)
                        self.planning_runs += 1

                        if self.path: 
                             self.current_path_index = 0
                             self.last_path_length = len(self.path)
                             print(f"Nueva ruta encontrada con {len(self.path)} puntos.")
                        else:
                             print("No se encontró ruta.")
                        planning_counter = 0

                    if self.path:
                        self.follow_path(robot_pos, robot_orientation) # Si hay ruta, síguela.
                    else:
                        print("Sin ruta, explorando...")
                        self.set_speeds(SPEED * 0.5, SPEED * 0.5) # Si no, explora un poco.
            
            elif self.robot_state == RobotState.AVOIDING_OBSTACLE:
                # Maniobra de evasión forzada: gira durante un tiempo fijo para escapar.
                if self.evasion_counter > 0:
                    print(f"EVASIÓN: Girando... ({self.evasion_counter}/{ESCAPE_STEPS})")
                    self.set_speeds(EVASION_TURN_SPEED, -EVASION_TURN_SPEED) # Gira a la derecha
                    self.evasion_counter -= 1
                else:
                    # Cuando termina el giro, vuelve a la navegación normal.
                    self.robot_state = RobotState.NAVIGATING
                    print("-> Fin de la maniobra de evasión. Volviendo a NAVEGACIÓN.")
                    self.set_speeds(0, 0)
            
            elif self.robot_state == RobotState.GOAL_REACHED:
                # Ha llegado al destino. Imprime métricas y termina.
                navigation_time = self.robot.getTime() - start_time
                avg_planning_time_ms = (self.total_planning_time / self.planning_runs) * 1000 if self.planning_runs > 0 else 0
                explored_cells = np.count_nonzero(self.explored_grid)
                explored_percentage = (explored_cells / (GRID_SIZE * GRID_SIZE)) * 100

                print("\n" + "="*40)
                print("¡OBJETIVO FINAL ALCANZADO!")
                print("--- Métricas de Desempeño ---")
                print(f"Tiempo total de navegación: {navigation_time:.2f} segundos")
                print(f"Longitud del path (celdas): {self.last_path_length}")
                print(f"Tiempo de planificación (A*): {avg_planning_time_ms:.4f} milisegundos")
                print(f"Porcentaje del mapa explorado: {explored_percentage:.2f} %")
                print("="*40 + "\n")
                self.set_speeds(0, 0); break
            
            # --- 3. ACTUALIZACIÓN: Guardar estado para el próximo ciclo ---
            self.prev_robot_pos = robot_pos
            
            # Imprimir información de depuración periódicamente
            debug_counter += 1
            if debug_counter > 15:
                print(f"Estado: {self.robot_state.name} | Coords: (X={robot_pos.x:.2f}, Y={robot_pos.y:.2f})")
                debug_counter = 0

            # Comprobar si se ha alcanzado físicamente el objetivo final.
            dist_to_goal = math.sqrt((self.goal_pos.x - robot_pos.x)**2 + (self.goal_pos.y - robot_pos.y)**2)
            if dist_to_goal < GOAL_THRESHOLD: self.robot_state = RobotState.GOAL_REACHED

if __name__ == "__main__":
    navigator = RobotNavigator()
    navigator.run()
