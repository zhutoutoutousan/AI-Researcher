from typing import Dict
from research_agent.inno.util import function_to_json
from research_agent.inno.types import Result
import json
from research_agent.inno.registry import register_tool

@register_tool("plan_dataset")
def plan_dataset(dataset_description: str, dataset_location: str, task_definition: str, read_data_step: str, data_preprocessing_step: str, data_dataloader_step: str):
    """
    Plan the dataset for the task. Use this tool after you have carefully reviewed the existing resources and understand the task.

    Args: 
        dataset_description: The description of the dataset.
        dataset_location: The location of the dataset.
        task_definition: The definition of the task.
        read_data_step: The description of how to read data (with code if necessary)
        data_preprocessing_step: The description of how to preprocess the data (with code if necessary)
        data_dataloader_step: The description of how to create dataloader for the data (with code if necessary)
    
    Returns:
        A Result object that contains the dataset plan.
    """
    dataset_plan = {
        "dataset_description": dataset_description,
        "dataset_location": dataset_location,
        "task_definition": task_definition,
        "data_processing": {
            "read_data": read_data_step,
            "data_preprocessing": data_preprocessing_step,
            "data_dataloader": data_dataloader_step
        }
    }
    return Result(
        value=f"""\
I have planned the dataset for the task. Here is the plan:
{json.dumps(dataset_plan, ensure_ascii=False, indent=4)}
        """, 
        context_variables={"dataset_plan": dataset_plan}
    )

@register_tool("plan_model")
def plan_model(model_input: str, model_output: str, key_components: list[str]):
    """
    Plan the model for the task. Use this tool after you have carefully reviewed the existing resources and understand the task.

    Args:
        model_input: The input of the model. You should describe the input in detail, including the type, shape, and other properties.
        model_output: The output of the model. You should describe the output in detail, including the type, shape, and other properties.
        key_components: A list of descriptions of the key components of the model (with code if necessary)
    
    Returns:
        A Result object that contains the model plan.
    """
    model_plan = {
        "model_input": model_input,
        "model_output": model_output,
        "key_components": key_components
    }
    return Result(
        value=f"""\
I have planned the model for the task. Here is the plan:
{json.dumps(model_plan, ensure_ascii=False, indent=4)}
        """, 
        context_variables={"model_plan": model_plan}
    )

@register_tool("plan_training")
def plan_training(training_pipeline: str, loss_function: str, optimizer: str, training_configurations: str, monitor_and_logging: str):
    """
    Plan the training process for the model. Use this tool after you have carefully reviewed the existing resources and understand the task.

    Args:
        training_pipeline: The pipeline of the training process. (with code if necessary)
        loss_function: The loss function of the model. (with code if necessary)
        optimizer: The optimizer of the model. (with code if necessary)
        training_configurations: The configurations of the training process. (with code if necessary)
        monitor_and_logging: The monitor and logging of the training process. (with code if necessary)
    Returns:
        A Result object that contains the training plan.
    """
    training_plan = {
        "training_pipeline": training_pipeline,
        "loss_function": loss_function,
        "optimizer": optimizer,
        "training_configurations": training_configurations,
        "monitor_and_logging": monitor_and_logging
    }       
    return Result(
        value=f"""\
I have planned the training process for the model. Here is the plan:
{json.dumps(training_plan, ensure_ascii=False, indent=4)}
        """, 
        context_variables={"training_plan": training_plan}
    )

@register_tool("plan_testing")
def plan_testing(test_metric: str, test_data: str, test_code: str):
    """
    Plan the test process for the model. Use this tool after you have carefully reviewed the existing resources and understand the task.

    Args:
        test_metric: The metric used to evaluate the model. (with code if necessary)
        test_data: The data used to test the model. (with code if necessary)
        test_code: The code used to test the model. (with code if necessary)
    
    Returns:
        A Result object that contains the test plan.
    """
    testing_plan = {
        "test_metric": test_metric,
        "test_data": test_data,
        "test_function": test_code
    }
    return Result(
        value=f"""\
I have planned the test process for the model. Here is the plan:
{json.dumps(testing_plan, ensure_ascii=False, indent=4)}
        """, 
        context_variables={"testing_plan": testing_plan}
    )

if __name__ == "__main__":
    print(function_to_json(plan_dataset))
    print(function_to_json(plan_model))
    print(function_to_json(plan_training))
    print(function_to_json(plan_testing))