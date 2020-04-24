echo "/* GENERATED FILE - DO NOT EDIT */" > generated_weights.c
python3 onnx_to_c_code.py ../code/IL_neighbors_rho.onnx weights_n_rho >> generated_weights.c
python3 onnx_to_c_code.py ../code/IL_neighbors_phi.onnx weights_n_phi >> generated_weights.c
python3 onnx_to_c_code.py ../code/IL_obstacles_rho.onnx weights_o_rho >> generated_weights.c
python3 onnx_to_c_code.py ../code/IL_obstacles_phi.onnx weights_o_phi >> generated_weights.c
python3 onnx_to_c_code.py ../code/IL_psi.onnx weights_psi >> generated_weights.c