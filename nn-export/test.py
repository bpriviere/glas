import nnexport

nnexport.nn_reset()
nnexport.nn_add_neighbor([1,2])
nnexport.nn_add_neighbor([3,4])
nnexport.nn_add_obstacle([1,2])
print(nnexport.nn_eval([0,0]))
