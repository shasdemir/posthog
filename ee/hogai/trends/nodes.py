from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from ee.hogai.schema_generator.nodes import BaseMessage, HumanMessagePromptTemplate, LangchainAssistantMessage, SchemaGeneratorNode, SchemaGeneratorToolsNode, merge_message_runs
from ee.hogai.schema_generator.utils import SchemaGeneratorOutput
from ee.hogai.taxonomy_agent.nodes import TaxonomyAgentPlannerNode, TaxonomyAgentPlannerToolsNode
from ee.hogai.trends.prompts import (
    react_system_prompt,
    trends_system_prompt,
)
from ee.hogai.trends.toolkit import TRENDS_SCHEMA, TrendsTaxonomyAgentToolkit
from ee.hogai.utils import AssistantNode, AssistantNodeName, AssistantState
from posthog.schema import AssistantMessage, AssistantTrendsQuery, VisualizationMessage


class TrendsPlannerNode(TaxonomyAgentPlannerNode):
    def run(self, state: AssistantState, config: RunnableConfig) -> AssistantState:
        toolkit = TrendsTaxonomyAgentToolkit(self._team)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", react_system_prompt),
            ],
            template_format="mustache",
        )
        return super()._run(state, prompt, toolkit, config=config)


class TrendsPlannerToolsNode(TaxonomyAgentPlannerToolsNode):
    def run(self, state: AssistantState, config: RunnableConfig) -> AssistantState:
        toolkit = TrendsTaxonomyAgentToolkit(self._team)
        return super()._run(state, toolkit, config=config)


TrendsSchemaGeneratorOutput = SchemaGeneratorOutput[AssistantTrendsQuery]


class TrendsGeneratorNode(SchemaGeneratorNode[AssistantTrendsQuery]):
    insight_name = "Trends"
    output_model = TrendsSchemaGeneratorOutput

    def run(self, state: AssistantState, config: RunnableConfig) -> AssistantState:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", trends_system_prompt),
            ],
            template_format="mustache",
        )
        return super()._run(state, prompt, config=config)

    @property
    def _model(self):
        return ChatOpenAI(model="gpt-4o", temperature=0.2, streaming=True).with_structured_output(
            TRENDS_SCHEMA,
            method="function_calling",
            include_raw=False,
        )


class TrendsGeneratorToolsNode(SchemaGeneratorToolsNode):
    pass

class SummarizeTrendsNode(AssistantNode):
    name = AssistantNodeName.SUMMARIZE_TRENDS

    def run(self, state: AssistantState, config: RunnableConfig):
        if not isinstance(VisualizationMessage, state["messages"][-1]):
            raise Exception()

        trends_summarization_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Act as an expert product manager. Your task is to summarize query results in a a concise way.
Offer actionable feedback if possible with your context.

The product being analyzed is described as follows:
{{product_description}}""",
                ),
            ],
            template_format="mustache",
        ) + self._reconstruct_conversation(state)
        merger = merge_message_runs()

        chain = trends_summarization_prompt | merger | self._model

        message = chain.invoke(
            {
                "product_description": self._team.project.product_description,
            },
            config,
        )

        return {"messages": [AssistantMessage(content=str(message.content))]}

    @property
    def _model(self):
        return ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)  # Slightly higher temp than earlier steps

    def _reconstruct_conversation(
        self, state: AssistantState, validation_error_message: Optional1[str] = None
    ) -> list[BaseMessage]:
        """
        Reconstruct the conversation for the generation. Take all previously generated questions, plans, and schemas, and return the history.
        """
        messages = state.get("messages", [])

        conversation: list[BaseMessage] = []

        for message in messages:
            if message.type == "human":
                conversation.append(
                    HumanMessagePromptTemplate.from_template(trends_question_prompt, template_format="mustache").format(
                        question=message.content
                    )
                )
            elif message.type == "ai":
                conversation.append(LangchainAssistantMessage(content=message.content))
            elif message.type == "ai/failure":
                conversation.append(
                    LangchainAssistantMessage(content="Something went wrong while answering.")  # TODO: Better message
                )
            elif message.type == "ai/viz":
                conversation.append(
                    LangchainAssistantMessage(content=message.answer.model_dump_json() if message.answer else "")
                )
            else:
                raise ValueError(f"Unknown message type: {message.type}")

        conversation.append(
            HumanMessagePromptTemplate.from_template("Summarize these results.", template_format="mustache").format()
        )

        return conversation
