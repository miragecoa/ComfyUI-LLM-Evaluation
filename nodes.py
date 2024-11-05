from comfyui import Node


class LoadFileNode(Node):
    def __init__(self):
        # Set the category to "LLM Evaluation"
        super().__init__("Load File", category="LLM Evaluation")

        # Define the input for the filename
        self.add_input("filename", Node.InputType.STRING)

        # Define the output for the file content
        self.add_output("file_text", Node.OutputType.STRING)

    def process(self):
        # Get the filename from the input
        filename = self.get_input_value("filename")

        try:
            # Read the file content
            with open(filename, 'r', encoding='utf-8') as file:
                file_text = file.read()

            # Output the file text
            self.set_output_value("file_text", file_text)
        except FileNotFoundError:
            # Handle the case where the file is not found
            self.set_output_value("file_text", f"Error: File '{filename}' not found.")
        except Exception as e:
            # Handle any other exceptions
            self.set_output_value("file_text", f"Error reading file: {e}")


def register():
    Node.register(LoadFileNode)
