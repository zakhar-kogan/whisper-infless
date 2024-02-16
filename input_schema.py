INPUT_SCHEMA = {
    "audio_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["http://thepodcastexchange.ca/s/Porsche-Macan-July-5-2018-1.mp3"]
    },
    "timestamps": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [True]
    }
}
