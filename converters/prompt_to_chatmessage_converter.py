from typing import List, Dict

from haystack import component
from haystack.dataclasses import ChatMessage


@component
class PromptToChatMessage:
    def __init__(self, prompt: str, message_list: List[ChatMessage]):
        """
        Initialize the PromptToChatMessageConverter component.

        :param prompt:
            The prompt to convert to a chat message.
        """
        self.prompt = prompt
        self.message_list = message_list

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "prompt": self.prompt,
            "message_list": [message.to_dict() for message in self.message_list]
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        return cls(
            prompt=data["prompt"],
            message_list=[ChatMessage.from_dict(message) for message in data["message_list"]]
        )

    @component.output_types(chat_messages=List[ChatMessage])
    def run(self, prompt: str) -> Dict[str, List[ChatMessage]]:
        """
        Convert a prompt to a chat message.

        :param prompt: The prompt to convert to a chat message.
        :returns: A list of chat messages.
        """
        self.message_list.append(ChatMessage.from_system(content=prompt))
        return {"chat_messages": self.message_list}
