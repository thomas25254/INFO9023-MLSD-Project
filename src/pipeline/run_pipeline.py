from config import LOCATION, PIPELINE_ROOT, PROJECT_ID
from data_preparation import data_preparation
from evaluation import evaluation
from google.cloud import aiplatform
from kfp import compiler, dsl
from threshold import compute_threshold
from training import training


@dsl.pipeline(name="hearedit-pipeline", pipeline_root=PIPELINE_ROOT)
def hearedit_pipeline(
    gcs_dataset_uri: str = "gs://mlops-2026-dataset-bucket/train-clean-100/train-clean-100/",
    gcs_threshold_output_uri: str = "gs://hearedit-models/artifacts/threshold_pipeline.json",
    epochs: int = 5,
    batch_size: int = 8,
):
    # Etape 1 - Data preparation sur CPU
    data_prep_task = data_preparation(
        gcs_dataset_uri=gcs_dataset_uri,
    )

    # Etape 2 - Training sur GPU
    training_task = training(
        train_split=data_prep_task.outputs["train_split"],
        val_split=data_prep_task.outputs["val_split"],
        gcs_dataset_uri=gcs_dataset_uri,
        epochs=epochs,
        batch_size=batch_size,
    )
    training_task.set_accelerator_type("NVIDIA_TESLA_T4")
    training_task.set_accelerator_limit(1)
    training_task.set_memory_limit("15G")
    training_task.set_cpu_limit("4")
    # Etape 3 - Evaluation sur CPU
    evaluation_task = evaluation(
        test_split=data_prep_task.outputs["test_split"],
        model=training_task.outputs["model"],
        gcs_dataset_uri=gcs_dataset_uri,
    )

    # Etape 4 - Threshold sur CPU
    threshold_task = compute_threshold(
        gcs_dataset_uri=gcs_dataset_uri,
        gcs_threshold_output_uri=gcs_threshold_output_uri,
        model=training_task.outputs["model"],
    ).after(evaluation_task)
    threshold_task.set_memory_limit("16G")


if __name__ == "__main__":
    compiler.Compiler().compile(hearedit_pipeline, "hearedit_pipeline.json")
    print("Pipeline compilé -> hearedit_pipeline.json")

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    job = aiplatform.PipelineJob(
        display_name="hearedit-pipeline",
        template_path="hearedit_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
    )
    job.submit()
    print("Pipeline soumis !")
