from deeplite_torch_zoo import get_eval_function, list_models

all_classification_models = list_models(task_type_filter='classification', print_table=False, return_list=True)

all_objectdetection_models = list_models(task_type_filter='object_detection', print_table=False, return_list=True)

all_segmentation_models = list_models(task_type_filter='semantic_segmentation', print_table=False, return_list=True)

no_eval_func = []

def get_function(all_models):

    for (model_name, dataset_name) in all_models:

        try:
            funct  = get_eval_function(model_name=model_name,dataset_name=dataset_name)
            #print(f'Evaluation function for {model_name} and {dataset_name} is {funct} \n')
        except:
            no_eval_func.append((model_name,dataset_name))
            #print(f'****Evaluation function for {model_name} and {dataset_name} do not exist \n')
    
    print(no_eval_func)
get_function(all_classification_models)
get_function(all_objectdetection_models)
get_function(all_segmentation_models)

#get_eval_function('deeplab_mobilenet', 'voc_20')

