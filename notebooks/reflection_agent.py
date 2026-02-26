import os
from groq import Groq
from dotenv import load_dotenv
from colorama import Fore, init

init()
load_dotenv()

class ReflectionAgent:
    def __init__(self, model = "llama-3.1-8b-instant"):
        self.client = Groq()
        self.model = model
    
    def generate(self, generation_chat_history: list, verbose: int = 0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=generation_chat_history
            )

            output = response.choices[0].message.content

            if not isinstance(output, str):
                raise TypeError("Model output is not a string")

            if verbose > 0:
                print(Fore.BLUE + "\n\nGENERATION\n\n" + output)

            return output

        except Exception as e:
            print(f"Generation error: {e}")
            return ""  
    
    def reflect(self, reflection_chat_history, verbose: int = 0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=reflection_chat_history
            )

            critique = response.choices[0].message.content

            if not isinstance(critique, str):
                raise TypeError("Critique output is not string")

            if verbose > 0:
                print(Fore.GREEN + "\n\nREFLECTION\n\n" + critique)

            return critique

        except Exception as e:
            print(f"Reflection error: {e}")
            return "" 

    def run(self, 
            generation_system_prompt: str,
            reflection_system_prompt: str,
            user_prompt: str,
            n_steps = 3,
            verbose: int = 0):
        """
        Runs reflection agent until the response is satisfied or until the number of turns defined
        
        """
        try:
            generation_history = [
                {
                    "role": "system",
                    "content": generation_system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            reflection_history = [
                {
                    "role": "system",
                    "content": reflection_system_prompt
                }
            ]
            for step in range(n_steps):

                print(f"=========== STEP: {step} ===========")

                generation = self.generate(
                    generation_chat_history=generation_history,
                    verbose=verbose
                )

                if not generation or not generation.strip():
                    print("Generation invalid. Stopping.")
                    break

                generation_history.append({
                    "role": "assistant",
                    "content": generation
                })

                reflection_history.append({
                    "role": "user",
                    "content": generation
                })

                critique = self.reflect(
                    reflection_chat_history=reflection_history,
                    verbose=verbose
                )

                if not critique or not critique.strip():
                    print("Critique invalid. Stopping.")
                    break

                reflection_history.append({
                    "role": "assistant",
                    "content": critique
                })

                generation_history.append({
                    "role": "user",
                    "content": critique
                })
            
            return generation
        except Exception as e:
            print(f"Exception occured in run {e}")


