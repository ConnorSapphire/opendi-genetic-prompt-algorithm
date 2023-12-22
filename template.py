import numpy as np

class Template:
    """Class to handle interacting with the template files.
    """
    
    def __init__(self, template = None, demo_template = None):
        """Initiate Template.

        Args:
            template (str, optional): File location of template. Defaults to None.
            demo_template (str, optional): File location of demo_template. Defaults to None.

        Raises:
            ValueError: No template file provided.
        """
        if template:
            try: 
                self.template = open(template).read()
                self.demo_template = open(demo_template).read()
            except FileNotFoundError: 
                print("File not found. Using the provided strings as templates")
                self.template = template
                self.demo_template = demo_template
        else:
            raise ValueError("Template file required")

    def populate_single_demo(self, demo):
        if self.demo_template:
            new_demo = self.demo_template.replace("[question]", demo["question"])
            new_demo = new_demo.replace("[answer]", demo["answer"])
            return new_demo
        return f"Question: {demo['question']}\nAnswer: {demo['answer']}"

    def populate_multiple_demos(self, demos):
        return "\n\n".join([self.populate_single_demo(demo) for demo in demos])

    def get_prompt(self, instruction, demos = [], question = "", context = ""):
        """Retrieve prompt from template
        """
        
        if len(demos) == 0:
            demo_string = ""
        else:
            demo_string = self.populate_multiple_demos(demos)
        ip = self.template
        ip = ip.replace("[instruction]", instruction).replace("[demos]", demo_string).replace("[question]", question).replace("[context]", context)
        return ip

    def get_llm_input(self, question = "", instruction = "", demos = [], context = ""):
        """Retrieve input for LLM.
        """
        
        return self.get_prompt(instruction, demos, question, context = context)
        # return f"{get_prompt(instruction, demos)} \nQuestion: {question}\nAnswer: "
