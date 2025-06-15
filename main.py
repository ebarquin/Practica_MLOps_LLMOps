import text_classification_pipelineipeline as func

def main():
    print("Ejecutamos el main")
    args_values = func.argumentos()
    x_train, x_test, y_train, y_test, target_names = func.load_and_prepare_data()
    func.mlflow_tracking(args_values.nombre_job, x_train, x_test, y_train, y_test, target_names, args_values.alpha_list)

if __name__ == "__main__":
    main()
