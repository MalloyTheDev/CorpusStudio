from pathlib import Path
from corpus_studio.storage.project import DatasetProject, create_project


def main():
    project = DatasetProject(
        id="example_instruction_project",
        name="Example Instruction Project",
        schema_id="instruction",
    )
    path = create_project(Path("data/projects"), project)
    print(f"Created project at {path}")


if __name__ == "__main__":
    main()
