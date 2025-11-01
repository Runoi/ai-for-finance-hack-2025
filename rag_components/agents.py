class SeaAgent:
    def __init__(self, llm_client, prompt_template):
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    def analyze(self, question, context):
        print(">> SeaAgent.analyze() вызван (пока не реализован)")
        # В Sprint 2 здесь будет логика аудита
        return {"is_sufficient": True} 

class RefinementAgent:
    def __init__(self, llm_client, prompt_template):
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        
    def refine(self, question, analysis_summary):
        print(">> RefinementAgent.refine() вызван (пока не реализован)")
        # В Sprint 2 здесь будет логика уточнения запросов
        return []