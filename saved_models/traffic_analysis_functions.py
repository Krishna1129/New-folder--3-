
# Traffic Analysis Functions
# This script contains all the functions for vehicle detection and traffic timing analysis

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import yaml

def calculate_green_time(car_count, motorcycle_count, base_time=10, time_per_car=4.5, time_per_motorcycle=1.5, min_time=15, max_time=60):
    """Calculate green signal time based on car and motorcycle counts, with min and max cap."""
    green_time = base_time + (car_count * time_per_car) + (motorcycle_count * time_per_motorcycle)
    green_time = max(min_time, min(green_time, max_time))
    return green_time

def calculate_waiting_times(recommended_times):
    """Calculate waiting times for each lane in one cycle."""
    waiting_times = []
    for i in range(len(recommended_times)):
        waiting_time = sum(recommended_times) - recommended_times[i]
        waiting_times.append(waiting_time)
    return waiting_times

def compare_with_constant(recommended_times, constant_wait_time=180):
    """Compare AI recommendations with constant waiting time."""
    ai_avg_wait_time = sum(recommended_times) * 3 / 4
    time_saved = constant_wait_time - ai_avg_wait_time
    improvement_percentage = (time_saved / constant_wait_time) * 100
    return ai_avg_wait_time, time_saved, improvement_percentage

# Load the trained model
model = YOLO('saved_models/vehicle_detector_with_timing.pt')

# Example usage:
# recommended_times = [31.0, 26.5, 23.5, 28.0]  # Example times
# waiting_times = calculate_waiting_times(recommended_times)
# ai_avg, saved, improvement = compare_with_constant(recommended_times)
