import math
import time
import numpy as np
import random
import copy
import json
import re
from .model import PlacementProcedure, BRKGA

container_options = {
    "BLIND_VAN": (255, 146, 130),
    "CDE": (350, 160, 160)
}

def run_layouting_algorithm(shipment_data, selected_container):
    print("Running BRKGA algorithm...")
    print(shipment_data)

    boxes = []
    box_DO_map = []

    def extract_do_number(do_num):
        match = re.search(r'\d+', do_num)
        return int(match.group()) if match else 0

    sorted_DOs = sorted(shipment_data.keys(), key=lambda x: extract_do_number(x), reverse=True)

    for idx, do in enumerate(sorted_DOs):
        for box_id, dims in shipment_data[do].items():
            boxes.append(tuple(dims))
            box_DO_map.append(idx)

    inputs = {
        'v': boxes,
        'V': [container_options[selected_container]],
        'box_DO_map': box_DO_map,
        'DO_count': len(sorted_DOs),
        'DOs_num': sorted_DOs
    }

    model = BRKGA(inputs, num_generations=30, num_individuals=40, num_elites=5, num_mutants=3, eliteCProb=0.7)
    model.fit(patient=10, verbose=False)

    inputs['solution'] = model.solution
    decoder = PlacementProcedure(inputs, model.solution)
    fitness = decoder.evaluate()

    # Jika tidak cukup di satu container, coba container lebih besar
    if math.floor(fitness) > 1:
        selected_container = "CDE"
        inputs['V'] = [container_options[selected_container]]

        model = BRKGA(inputs, num_generations=30, num_individuals=40, num_elites=5, num_mutants=3, eliteCProb=0.7)
        model.fit(patient=10, verbose=False)

        inputs['solution'] = model.solution
        decoder = PlacementProcedure(inputs, model.solution)
        fitness = decoder.evaluate()

    output_layout = []
    DOs_list = inputs['DOs_num']
    for i, box_data in enumerate(decoder.Bins[0].load_items):
        min_corner, max_corner, do_index = box_data
        do_num = DOs_list[do_index]
        output_layout.append({
            "do_num": do_num,
            "box_id": i + 1,
            "do_index": int(do_index),
            "min_corner": [float(coord) for coord in min_corner],
            "max_corner": [float(coord) for coord in max_corner]
        })

    return {
        "fitness": fitness,
        "layout": output_layout,
        "selected_container": selected_container
    }
