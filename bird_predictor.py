import joblib
import librosa
import librosa.display
import numpy as np
from os import scandir
import os
import streamlit as st
import matplotlib.pyplot as plt
from pydub import AudioSegment
from PIL import Image

num_dict = {'0': 'Alder Flycatcher', '1': 'American Avocet', '2': 'American Bittern', '3': 'American Crow', '4': 'American Goldfinch', '5': 'American Kestrel', '6': 'Buff-bellied Pipit', '7': 'American Redstart', '8': 'American Robin', '9': 'American Wigeon', '10': 'American Woodcock', '11': 'American Tree Sparrow', '12': "Anna's Hummingbird", '13': 'Ash-throated Flycatcher', '14': "Baird's Sandpiper", '15': 'Bald Eagle', '16': 'Baltimore Oriole', '17': 'Sand Martin', '18': 'Barn Swallow', '19': 'Black-and-White Warbler', '20': 'Belted Kingfisher', '21': 'Sage Sparrow', '22': "Bewick's Wren", '23': 'Black-Billed Cuckoo', '24': 'Black-Billed Magpie', '25': 'Blackburnian Warbler', '26': 'Black-Capped Chickadee', '27': 'Black-Chinned Hummingbird', '28': 'Black-Headed Grosbeak', '29': 'Blackpoll Warbler', '30': 'Black-Throated Sparrow', '31': 'Black Phoebe', '32': 'Blue Brosbeak', '33': 'Blue Jay', '34': 'Brown-Headed Cowbird', '35': 'Bobolink', '36': "Bonaparte's Gull", '37': 'Northern Barred Owl', '38': "Brewer's Blackbird", '39': "Brewer's Sparrow", '40': 'Brown Creeper', '41': 'Brown Thrasher', '42': 'Broad-Tailed Hummingbird', '43': 'Broad-Winged Hawk', '44':'Black-Throated Blue Warbler', '45': 'Black-Throated Green Warbler', '46': 'Black-Throated Grey Warbler', '47': 'Bufflehead', '48': 'Blue-Grey Gnatcatcher', '49': 'Blue-Headed Vireo', '50': "Bullock's Oriole", '51': 'American Bushtit', '52': 'Blue-Winged Teal', '53': 'Blue-Winged Warbler', '54': 'Cactus Wren', '55': 'California Gull', '56': 'calqua', '57': 'camwar', '58': 'cangoo', '59': 'canwar', '60': 'canwre', '61': 'carwre', '62': 'casfin', '63': 'caster1', '64': 'casvir', '65': 'cedwax', '66': 'chispa', '67': 'chiswi', '68': 'chswar', '69': 'chukar', '70': 'clanut', '71': 'cliswa', '72': 'comgol', '73': 'comgra', '74': 'comloo', '75': 'commer', '76': 'comnig', '77': 'comrav', '78': 'comred', '79': 'comter', '80': 'comyel', '81': 'coohaw', '82': 'coshum', '83': 'cowscj1', '84': 'daejun', '85': 'doccor', '86': 'dowwoo', '87': 'dusfly', '88':'eargre', '89': 'easblu', '90': 'easkin', '91': 'easmea', '92': 'easpho', '93': 'eastow', '94': 'eawpew', '95': 'eucdov', '96': 'eursta', '97': 'evegro', '98': 'fiespa', '99': 'fiscro', '100': 'foxspa', '101': 'gadwal', '102': 'gcrfin', '103': 'gnttow', '104': 'gnwtea', '105': 'gockin', '106': 'gocspa', '107': 'goleag', '108': 'grbher3', '109': 'grcfly', '110': 'greegr', '111': 'greroa', '112': 'greyel', '113': 'grhowl', '114': 'grnher', '115': 'grtgra', '116': 'grycat', '117': 'gryfly', '118': 'haiwoo', '119': 'hamfly', '120': 'hergul', '121': 'herthr', '122': 'hoomer', '123': 'hoowar', '124': 'horgre', '125': 'horlar', '126': 'houfin', '127': 'houspa', '128': 'houwre', '129': 'indbun', '130':'juntit1', '131': 'killde', '132': 'labwoo', '133': 'larspa', '134': 'lazbun', '135': 'leabit', '136': 'leafly', '137': 'leasan', '138': 'lecthr', '139': 'lesgol', '140': 'lesnig', '141': 'lesyel', '142': 'lewwoo', '143': 'linspa', '144': 'lobcur', '145': 'lobdow', '146': 'logshr', '147': 'lotduc', '148': 'louwat', '149': 'macwar', '150': 'magwar', '151': 'mallar3', '152': 'marwre', '153': 'merlin', '154': 'moublu', '155': 'mouchi', '156': 'moudov', '157': 'norcar','158': 'norfli', '159': 'norhar2', '160': 'normoc', '161': 'norpar', '162': 'norpin', '163': 'norsho', '164': 'norwat', '165': 'nrwswa', '166': 'nutwoo', '167': 'olsfly', '168': 'orcwar', '169': 'osprey', '170': 'ovenbi1', '171': 'palwar', '172': 'pasfly', '173': 'pecsan', '174': 'perfal', '175': 'phaino', '176': 'pibgre', '177': 'pilwoo', '178': 'pingro', '179': 'pinjay', '180': 'pinsis', '181': 'pinwar', '182': 'plsvir', '183': 'prawar', '184': 'purfin', '185': 'pygnut', '186': 'rebmer', '187': 'rebnut', '188': 'rebsap', '189': 'rebwoo', '190': 'redcro', '191': 'redhea', '192': 'reevir1', '193': 'renpha', '194': 'reshaw', '195': 'rethaw', '196': 'rewbla', '197': 'ribgul', '198': 'rinduc', '199': 'robgro', '200': 'rocpig', '201': 'rocwre', '202': 'rthhum', '203': 'ruckin', '204': 'rudduc', '205': 'rufgro', '206': 'rufhum', '207': 'rusbla', '208': 'sagspa1', '209': 'sagthr', '210': 'savspa', '211': 'saypho', '212': 'scatan','213': 'scoori', '214': 'semplo', '215': 'semsan', '216': 'sheowl', '217': 'shshaw', '218': 'snobun', '219': 'snogoo', '220': 'solsan', '221': 'sonspa', '222': 'sora', '223': 'sposan', '224': 'spotow', '225': 'stejay', '226': 'swahaw','227': 'swaspa', '228': 'swathr', '229': 'treswa', '230': 'truswa', '231': 'tuftit', '232': 'tunswa', '233': 'veery', '234': 'vesspa', '235': 'vigswa', '236': 'warvir', '237': 'wesblu', '238': 'wesgre', '239': 'weskin', '240': 'wesmea', '241': 'wessan', '242': 'westan', '243': 'wewpew', '244': 'whbnut', '245': 'whcspa', '246': 'whfibi', '247': 'whtspa', '248': 'whtswi', '249': 'wilfly', '250': 'wilsni1', '251': 'wiltur', '252': 'winwre3', '253': 'wlswar', '254': 'wooduc', '255': 'wooscj2', '256': 'woothr', '257': 'y00475', '258': 'yebfly', '259': 'yebsap', '260': 'yehbla', '261': 'yelwar', '262': 'yerwar', '263': 'yetvir'}
name_dict = {'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4, 'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9, 'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14, 'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19, 'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24, 'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29, 'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33,'bnhcow': 34, 'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39, 'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44, 'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49, 'bulori':50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54, 'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59, 'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64, 'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69, 'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74, 'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79, 'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84, 'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89, 'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94, 'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99, 'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104, 'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109, 'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114, 'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119, 'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124, 'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129, 'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134, 'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139, 'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144, 'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149, 'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154, 'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159, 'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164, 'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169, 'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174, 'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179, 'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184, 'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189, 'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194, 'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199, 'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204, 'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209, 'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214, 'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219, 'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224, 'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229, 'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234, 'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239, 'wesmea': 240, 'wessan':241, 'westan': 242, 'wewpew': 243, 'whbnut': 244, 'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249, 'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254, 'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259, 'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263}
N_MFCC = 40

pic_dict = {str(i): './images/' + num_dict[str(i)] + '.jpg' for i in range(264)}

@st.cache
def load_model():
    model = joblib.load('bird_model.pkl')
    return model

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=8000, n_mfcc=N_MFCC)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed

def make_prediction(song):
    data = extract_features(song)
    prediction = model.predict([data])[0]
    return prediction

def display_spect(audio):
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig = plt.figure()
    librosa.display.specshow(S_db)
    return fig

def play_song(file_name):
    audio = open(file_name, 'rb')
    audio_b = audio.read()
    st.audio(audio_b, format='audio/mp3')

def show_bird(prediction):
    try: 
        img = Image.open(pic_dict[str(prediction)])
        st.image(img, use_column_width=True, caption='your lovely ' + num_dict[str(prediction)])
    except FileNotFoundError:
        st.write('no image available for your lovely ' + num_dict[str(prediction)])

st.title("Birdcall Identifier")

st.header("Trained on over 21,000 audio files, I'll classify your birdcall into one of 263 species.")

model = load_model()

song = st.file_uploader("Upload an mp3: ", type=['mp3'])

if song is not None:
    song_file = AudioSegment.from_mp3(song)
    path = './' + song.name
    song_file.export(path, format='mp3')

    audio, sample_rate = librosa.load(path, sr=8000, res_type='kaiser_fast')
    prediction = make_prediction(audio)
    st.write('We think ' + song.name + ' is the call of the ' + '***' + num_dict[str(prediction)]+ '***')

    
    play_song(path)
    pressed = st.checkbox('Show Spectrogram?')
    if pressed:
        fig = display_spect(audio)
        st.pyplot(fig)
    show_bird(prediction)

    os.remove(path)
