import discord
from discord.ext import commands
import wandb
from keras.models import load_model
import tensorflow
import re
import string
import pickle
import os
from dotenv import load_dotenv

load_dotenv('./.env')

wandb.login(key=os.getenv('WANDB_API_KEY'))
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

#load model from wandb
run = wandb.init(project='MichelLaPolice', entity='0xasept')
artifact = run.use_artifact('0xasept/MichelLaPolice/Matraque:PROD', type='model')
artifact_dir_toxic = artifact.download()
toxic_path = os.path.join(artifact_dir_toxic, 'model.h5')
model = load_model(toxic_path)


# prepare bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', description='A bot that say if the text  is toxic.', intents=intents)

def custom_standardization(sentence):
    sample = tensorflow.strings.lower(sentence)
    sample = tensorflow.strings.regex_replace(sample, '\W', ' ')
    sample = tensorflow.strings.regex_replace(sample, '\d', ' ')
    return tensorflow.strings.regex_replace(sample,
                         '[%s]'%re.escape(string.punctuation), '')

max_features = 10000
sequence_length = 250

vectorize_layer = tensorflow.keras.layers.TextVectorization(
                        standardize=custom_standardization,
                        split='whitespace',
                        max_tokens=max_features,
                        output_mode='int',
                        output_sequence_length=sequence_length,
                        encoding='utf-8')

# Load the vocabulary
artifact = run.use_artifact('0xasept/MichelLaPolice/Vocab:latest', type='model')
artifact_dir_vocab = artifact.download()
vocab_path = os.path.join(artifact_dir_vocab, 'vectorize_layer_vocab.pkl')
vocab = pickle.load(open(vocab_path, 'rb'))
# Update the vectorize_layer with the loaded vocabulary
vectorize_layer.set_vocabulary(vocab)


@bot.event
async def on_ready():
    print('Logged in as {0.user}'.format(bot))


@bot.command()
async def is_toxic(ctx, *, prompt):
    print("prompt asked : ", prompt)
    prompt_pre = vectorize_layer([prompt])
    result = model.predict(prompt_pre)
    
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Parcourir chaque classe de toxicité et afficher son label et sa probabilité associée
    response = ""
    for i, label in enumerate(labels):
        percentage = result[0][i] * 100
        response += f"{label.capitalize()}: {percentage:.2f}%\n"

    print("result : ", response)
    await ctx.send(response)

@bot.command()
async def update_model(ctx):
    run = wandb.init(project='MichelLaPolice', entity='0xasept')
    artifact = run.use_artifact('0xasept/MichelLaPolice/Matraque:PROD', type='model')
    artifact_dir_toxic = artifact.download()
    toxic_path = os.path.join(artifact_dir_toxic, 'model.h5')
    model = load_model(toxic_path)
    await ctx.send("Le modèle a été mis à jour !")



bot.run(DISCORD_TOKEN)