import math
import time
import numpy as np
import random
import copy
import json
import re
# from .model import PlacementProcedure, BRKGA
from .model_testing import PlacementProcedure, BRKGA

container_options = {
    "BLIND_VAN": (255, 146, 130),
    "CDE": (350, 160, 160)
}

def run_layouting_algorithm(shipment_data, selected_container, shipment_id, shipment_num):
    base_container = selected_container
    print("Running BRKGA algorithm...")
    print(shipment_data)

    boxes = []
    box_DO_map = []

    # Urutkan DO secara descending (prioritaskan DO dengan box besar/lebih banyak)
    # sorted_DOs = sorted(shipment_data.keys(), key=lambda x: -sum(d[0]*d[1]*d[2] for d in shipment_data[x].values()))
    # sorted_DOs = sorted(shipment_data.keys(), reverse=True)
    sorted_DOs = list(shipment_data.keys())[::-1]

    print("Sorted DOs: ",sorted_DOs)

    for do_idx, do in enumerate(sorted_DOs):
        for box_id, dims in shipment_data[do].items():
            length, width, height, quantity = dims
            for _ in range(quantity):
                boxes.append((length, width, height))
                box_DO_map.append(do_idx)

    print("Boxes: ",boxes)

    inputs = {
        'v': boxes,
        'V': [container_options[selected_container]],
        'box_DO_map': box_DO_map,
        'DO_count': len(sorted_DOs),
        'DOs_num': sorted_DOs
    }

    # Step 1: Jalankan algoritma untuk container awal
    model = BRKGA(inputs, 
             num_generations=20,  # Kurangi generasi
             num_individuals=30,  # Kurangi populasi
             num_elites=5, 
             num_mutants=5, 
             eliteCProb=0.8,      # Tingkatkan probabilitas elite
             multiProcess=True
        )
    model.fit(patient=10, verbose=False)
    decoder = PlacementProcedure(inputs, model.solution)
    fitness = decoder.evaluate()
    print("Tipe Truk: ", selected_container, "Fitness:", fitness)

    # Step 2: Jika BLIND_VAN dan fitness >= 2, switch ke CDE
    if selected_container == "BLIND_VAN" and math.floor(fitness) >= 2:
        print("BLIND_VAN tidak cukup, switching ke CDE...")
        selected_container = "CDE"
        inputs['V'] = [container_options[selected_container]]
        
        # Untuk CDE, tingkatkan parameter untuk memastikan solusi optimal
        model = BRKGA(inputs, 
                    num_generations=20,  # Kurangi generasi
                    num_individuals=30,  # Kurangi populasi
                    num_elites=5, 
                    num_mutants=5, 
                    eliteCProb=0.8,      # Tingkatkan probabilitas elite
                    multiProcess=True
             )
        model.fit(patient=15, verbose=False)
        decoder = PlacementProcedure(inputs, model.solution)
        fitness = decoder.evaluate()
        print("Tipe Truk: ", selected_container, "Fitness (2):", fitness)
        
        # Pastikan fitness CDE adalah 1.x (hanya butuh 1 container)
        if math.floor(fitness) > 1:
            # Jika masih tidak muat, paksa dengan menambah bin
            inputs['V'] = [container_options[selected_container]] * math.ceil(fitness)
            model = BRKGA(inputs, 
                    num_generations=20,  # Kurangi generasi
                    num_individuals=30,  # Kurangi populasi
                    num_elites=5, 
                    num_mutants=5, 
                    eliteCProb=0.8,      # Tingkatkan probabilitas elite
                    multiProcess=True
                    )
            model.fit(patient=10, verbose=False)
            decoder = PlacementProcedure(inputs, model.solution)
            fitness = decoder.evaluate()
            print("Tipe Truk: ", selected_container, "Fitness (3):", fitness)

        print("Tipe Truk: ", selected_container, "Fitness (4):", fitness)

    # Step 3: Format output
    # output_layout = []
    # for i, box_data in enumerate(decoder.Bins[0].load_items):
    #     min_corner, max_corner, do_index = box_data
    #     do_num = inputs['DOs_num'][do_index]
    #     output_layout.append({
    #         "do_num": do_num,
    #         "box_id": i + 1,
    #         "do_index": int(do_index),
    #         "min_corner": [float(coord) for coord in min_corner],
    #         "max_corner": [float(coord) for coord in max_corner]
    #     })

    output_layout = []
    for bin in decoder.Bins[:decoder.num_opend_bins]:
        for i, box_data in enumerate(bin.load_items):
            min_corner, max_corner, do_index = box_data
            do_num = inputs['DOs_num'][do_index]
            output_layout.append({
                "do_num": do_num,
                "box_id": i + 1,
                "do_index": int(do_index),
                "min_corner": [float(coord) for coord in min_corner],
                "max_corner": [float(coord) for coord in max_corner],
                # Tambahkan field urutan untuk sorting
                "do_priority": do_index
            })

    # Urutkan output berdasarkan urutan DO asli
    output_layout.sort(key=lambda x: x["do_priority"])

    return {
        "shipment_id": shipment_id,
        "shipment_num": shipment_num,
        "fitness": fitness,
        "base_container": base_container,
        "selected_container": selected_container,
        "layout": output_layout,
    }
