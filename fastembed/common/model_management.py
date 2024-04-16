from typing import List, Dict, Any


class ModelManagement:
    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        raise NotImplementedError()

    @classmethod
    def _get_model_description(cls, model_name: str) -> Dict[str, Any]:
        """
        Gets the model description from the model_name.

        Args:
            model_name (str): The name of the model.

        raises:
            ValueError: If the model_name is not supported.

        Returns:
            Dict[str, Any]: The model description.
        """
        for model in cls.list_supported_models():
            if model_name.lower() == model["model"].lower():
                return model

        raise ValueError(f"Model {model_name} is not supported in {cls.__name__}.")
