# UAV channel and environment model
# Customized and executed by Krish Chaudhari
# Focus: signal propagation, obstruction modeling, and covert communication metrics

import numpy as np
import random
from typing import List, Tuple, Dict
from scipy.stats import chi2


class Building:

    def __init__(self, x: float, y: float, width: float, height: float, building_height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.building_height = building_height
        
        # Precompute building boundaries for intersection checks
        self.x_min = x - width / 2
        self.x_max = x + width / 2
        self.y_min = y - height / 2
        self.y_max = y + height / 2
    
    def intersects_line(self, x1: float, y1: float, x2: float, y2: float) -> bool:

        # Liang-Barsky line clipping algorithm for line-rectangle intersection\
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return False
            
        t_min = 0.0
        t_max = 1.0
        
        # Check intersection along x-axis
        if dx != 0:
            t1 = (self.x_min - x1) / dx
            t2 = (self.x_max - x1) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        else:
            if x1 < self.x_min or x1 > self.x_max:
                return False
                
        # Check intersection along y-axis
        if dy != 0:
            t1 = (self.y_min - y1) / dy
            t2 = (self.y_max - y1) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        else:
            if y1 < self.y_min or y1 > self.y_max:
                return False
                
        return True


class ChannelModel:
    
    def __init__(self, channel_params, env_params=None, area_size_km: float = 1.0, seed: int = 42):

         # Validate required configuration parameters
         if channel_params is None:
             raise ValueError("channel_params is required and cannot be None")
         if env_params is None:
             raise ValueError("env_params is required and cannot be None")
         
        
         self.channel_params = channel_params
         self.env_params = env_params
         
         # Set random seed for reproducible experiments
         generation_range = self.env_params.get('building_generation_range', [0, 1000])
         range_min, range_max = generation_range
         self.area_size_m = range_max - range_min 
         self.area_size_km = self.area_size_m / 1000  
         
         np.random.seed(seed)
         random.seed(seed)
         
         # Generate synthetic building distribution
         self.buildings = self._generate_buildings()
    
    def _generate_buildings(self) -> List[Building]:
        buildings = []

        
        itu_beta = self.env_params.get('itu_beta', 40)
        density_factor = self.env_params.get('density_factor', 1.0)
        height_range = self.env_params.get('height_range', [10, 70])
        width_range = self.env_params.get('width_range', [40, 100])
        length_range = self.env_params.get('length_range', [40, 100])
        min_distance = self.env_params.get('min_distance', 5.0)
        max_attempts = self.env_params.get('max_attempts', 10)
        
       
        generation_range = self.env_params.get('building_generation_range', [0, 1000])
        range_min, range_max = generation_range
        
       
        base_num_buildings = int(itu_beta * (self.area_size_km ** 2))
        num_buildings = int(base_num_buildings * density_factor)
        
        for i in range(num_buildings):
            building_placed = False
            
            for attempt in range(max_attempts):
               
                width = np.random.uniform(*width_range)
                length = np.random.uniform(*length_range)

                
                x_min = max(range_min + width/2, width/2)
                x_max = min(range_max - width/2, self.area_size_m - width/2)
                y_min = max(range_min + length/2, length/2)
                y_max = min(range_max - length/2, self.area_size_m - length/2)

                
                if x_min >= x_max or y_min >= y_max:
                    continue
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                
               
                building_height = np.random.uniform(*height_range)
                
               
                candidate = Building(x, y, width, length, building_height)
                
               
                if self._is_building_valid(candidate, buildings, min_distance):
                    buildings.append(candidate)
                    building_placed = True
                    break
            
            # Skip placement if no valid position is found
        return buildings
    
    def _is_building_valid(self, candidate: Building, existing_buildings: List[Building], min_distance: float) -> bool:

       
        for existing in existing_buildings:
            
            dx = max(0, max(existing.x_min - candidate.x_max, candidate.x_min - existing.x_max))
            dy = max(0, max(existing.y_min - candidate.y_max, candidate.y_min - existing.y_max))
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                return False
        
      
        if self._is_in_warden_exclusion_zone(candidate):
            return False
        
        return True
    
    def _is_in_warden_exclusion_zone(self, candidate: Building) -> bool:

     
        warden_pos_range = self.env_params.get('warden_pos_range', [490, 510])
        exclusion_size = self.env_params.get('warden_exclusion_size', 100)
        
       
        warden_center_x = (warden_pos_range[0] + warden_pos_range[1]) / 2
        warden_center_y = (warden_pos_range[0] + warden_pos_range[1]) / 2
        
       
        exclusion_half_size = exclusion_size / 2
        exclusion_x_min = warden_center_x - exclusion_half_size
        exclusion_x_max = warden_center_x + exclusion_half_size
        exclusion_y_min = warden_center_y - exclusion_half_size
        exclusion_y_max = warden_center_y + exclusion_half_size
        
       
        if (candidate.x_max >= exclusion_x_min and candidate.x_min <= exclusion_x_max and
            candidate.y_max >= exclusion_y_min and candidate.y_min <= exclusion_y_max):
            return True
        
        return False
    
    def is_los(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> bool:

        # Check if signal path is blocked by any building
        for building in self.buildings:
            if building.intersects_line(uav_x, uav_y, ground_x, ground_y):
                
                # Simplified obstruction rule based on relative building height
                if building.building_height > self.channel_params['uav_height'] * 0.3:
                    return False
        return True
    
    def calculate_pathloss(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> float:

        # Compute 3D distance between UAV and ground node
        horizontal_dist = np.sqrt((uav_x - ground_x)**2 + (uav_y - ground_y)**2)
        distance_3d = np.sqrt(horizontal_dist**2 + self.channel_params['uav_height']**2)
        
      
        fc_ghz = self.channel_params['fc'] / 1e9
        
        # Determine Line-of-Sight or Non-Line-of-Sight link

        is_los_link = self.is_los(uav_x, uav_y, ground_x, ground_y)
        
        if is_los_link:
           
            pathloss = (self.channel_params['los_a'] * np.log10(distance_3d) + 
                       self.channel_params['los_b'] + 
                       self.channel_params['los_c'] * np.log10(fc_ghz))
        else:
          
            pathloss = (self.channel_params['nlos_a'] * np.log10(distance_3d) + 
                       self.channel_params['nlos_b'] + 
                       self.channel_params['nlos_c'] * np.log10(fc_ghz) + 
                       self.channel_params['nlos_d'] * horizontal_dist)
        
        return pathloss
    
    def calculate_snr(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> float:

        pathloss_db = self.calculate_pathloss(uav_x, uav_y, ground_x, ground_y)
        received_power_db = self.channel_params['tx_power'] - pathloss_db
        snr_db = received_power_db - self.channel_params['noise_power']
        return snr_db
    
    def calculate_transmission_rate(self, uav_x: float, uav_y: float, ground_x: float, ground_y: float) -> float:

        snr_db = self.calculate_snr(uav_x, uav_y, ground_x, ground_y)
        snr_linear = 10 ** (snr_db / 10)
        rate = np.log2(1 + snr_linear)
        return rate
    

    
    def calculate_dep(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        # Ray-assisted Detection Error Probability (DEP) computation
        return self._calculate_ray_assisted_dep(uav_x, uav_y, warden_x, warden_y)

    def _calculate_exact_dep(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        # Safety check: skip computation if warden is beyond sensing range
        max_ray_length = self.env_params['max_ray_length']
        distance_to_warden = np.sqrt((warden_x - uav_x)**2 + (warden_y - uav_y)**2)
        
        # If warden is outside ray range, communication is fully covert
        if distance_to_warden > max_ray_length:
            return 1.0
        
    
        P_u = 10 ** (self.channel_params['tx_power'] / 10)  
        iota = self.channel_params['noise_uncertainty']  
        iota_hat = self.channel_params['nominal_noise'] 
        
       
        pathloss_w = self.calculate_pathloss(uav_x, uav_y, warden_x, warden_y)
    
        received_power = P_u * (10 ** (-pathloss_w / 10))
        
    
        iota_linear = 10 ** (iota / 10)
        noise_upper = iota_linear * iota_hat
        noise_lower = iota_hat / iota_linear
        
        # Numerical stability checks
        if received_power <= 1e-12: 
            return 1.0
        
        if noise_upper <= noise_lower:
            return 0.0
        
        if received_power >= (noise_upper - noise_lower):
            return 0.0
        denominator = received_power + noise_lower
        if denominator <= 1e-12:
            return 1.0
        ratio = noise_upper / denominator
        if ratio <= 1.0:
            return 0.0

        try:
            log_ratio = np.log(ratio)
            log_base = 2 * iota * np.log(10) / 10 
            
            if log_base <= 1e-12:
                return 1.0
            
            dep = log_ratio / log_base
            return np.clip(dep, 0.0, 1.0)
        except (ValueError, OverflowError, ZeroDivisionError):
            # Fallback values for numerical edge cases
            return 1.0 if ratio > 1.0 else 0.0
    
    def is_covert_hole(self, uav_x: float, uav_y: float, warden_positions: List[Tuple[float, float]]) -> bool:
        return self._is_ray_assisted_covert_hole(uav_x, uav_y, warden_positions)
    
    def _calculate_ray_assisted_dep(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

        
        base_dep = self._calculate_exact_dep(uav_x, uav_y, warden_x, warden_y)
        
        
        obstruction_factor = self._calculate_ray_obstruction_factor(uav_x, uav_y, warden_x, warden_y)

      
        adjusted_dep = base_dep + obstruction_factor * (1.0 - base_dep) * self.channel_params.get('obstruction_weight', 0.3)
        
        return np.clip(adjusted_dep, 0.0, 1.0)
    
    def _calculate_ray_obstruction_factor(self, uav_x: float, uav_y: float, warden_x: float, warden_y: float) -> float:

    
        num_rays = self.env_params['num_rays']
        angle_step = 360.0 / num_rays
        max_ray_length = self.env_params['max_ray_length']
        
       
        dx_to_warden = warden_x - uav_x
        dy_to_warden = warden_y - uav_y
        distance_to_warden = np.sqrt(dx_to_warden**2 + dy_to_warden**2)
        
        if distance_to_warden < 1e-6:
            return 0.0 

        if distance_to_warden > max_ray_length:
            return 0.0
        
    
        warden_angle = np.degrees(np.arctan2(dy_to_warden, dx_to_warden))
        if warden_angle < 0:
            warden_angle += 360
        

        relevant_rays = []
        obstruction_count = 0
        

        for ray_idx in range(num_rays):
            ray_angle = ray_idx * angle_step
            angle_diff = abs(ray_angle - warden_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= 45: 
                relevant_rays.append(ray_idx)
                
             
                angle_rad = np.radians(ray_angle)
                dx = np.cos(angle_rad)
                dy = np.sin(angle_rad)
                
    
                ray_length = self._calculate_ray_building_intersection_distance(
                    uav_x, uav_y, dx, dy, max_ray_length)
                
             
                if ray_length < distance_to_warden:
                    obstruction_count += 1
        
        if not relevant_rays:
            return 0.0
        

        obstruction_ratio = obstruction_count / len(relevant_rays)
        
   
        distance_factor = min(1.0, distance_to_warden / max_ray_length)
        
        return obstruction_ratio * (1.0 - distance_factor * 0.5)
    
    def _calculate_ray_building_intersection_distance(self, start_x: float, start_y: float, 
                                                    dx: float, dy: float, max_length: float) -> float:

        min_distance = max_length
        
        for building in self.buildings:
          
            min_height = self.env_params.get('min_building_height', 40)
            if hasattr(building, 'building_height') and building.building_height <= min_height:
                continue
                
            x_min, x_max = building.x_min, building.x_max
            y_min, y_max = building.y_min, building.y_max
            
            intersections = []
            

            if abs(dx) > 1e-10:  
            
                t = (x_min - start_x) / dx
                if 0 < t < max_length:
                    y_intersect = start_y + t * dy
                    if y_min <= y_intersect <= y_max:
                        intersections.append(t)
                
               
                t = (x_max - start_x) / dx
                if 0 < t < max_length:
                    y_intersect = start_y + t * dy
                    if y_min <= y_intersect <= y_max:
                        intersections.append(t)
            
            if abs(dy) > 1e-10:  
            
                t = (y_min - start_y) / dy
                if 0 < t < max_length:
                    x_intersect = start_x + t * dx
                    if x_min <= x_intersect <= x_max:
                        intersections.append(t)
                
            
                t = (y_max - start_y) / dy
                if 0 < t < max_length:
                    x_intersect = start_x + t * dx
                    if x_min <= x_intersect <= x_max:
                        intersections.append(t)
            
            if intersections:
                min_distance = min(min_distance, min(intersections))
        
        return min_distance

    def calculate_ray_endpoints(self, start_x: float, start_y: float, map_size: float) -> List[Tuple[float, float]]:

        num_rays = self.env_params['num_rays']
        angle_step = 360.0 / num_rays
        max_ray_length = self.env_params['max_ray_length'] 
        
        ray_endpoints = []
        
        for ray_idx in range(num_rays):
            angle_deg = ray_idx * angle_step
            angle_rad = np.radians(angle_deg)
            
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
            
            
            building_distance = self._calculate_ray_building_intersection_distance(
                start_x, start_y, dx, dy, max_ray_length)
            
          
            ray_length = min(max_ray_length, building_distance)
            
     
            end_x = start_x + ray_length * dx
            end_y = start_y + ray_length * dy
            ray_endpoints.append((end_x, end_y))
        
        return ray_endpoints

    def _is_ray_assisted_covert_hole(self, uav_x: float, uav_y: float, 
                                   warden_positions: List[Tuple[float, float]]) -> bool:

        covert_threshold = self.env_params['covertness_threshold']
        
        for warden_x, warden_y in warden_positions:
      
            dep = self._calculate_ray_assisted_dep(uav_x, uav_y, warden_x, warden_y)
            
         
            if dep <= covert_threshold:
                return False
        
        
        return True