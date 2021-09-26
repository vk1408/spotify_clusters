import seaborn as sns
import pandas as pd
import random

features_limits = {'danceability':[0.0,1.0], 
                   'energy':[0.0,1.0], 
                   'loudness':[-60.0,0.0], 
                   'key':[0,11],
                   'mode':[0,1], 
                   'speechiness':[0.0,1.0],
                   'acousticness':[0.0,1.0], 
                   'instrumentalness':[0.0,1.0], 
                   'liveness':[0.0,1.0], 
                   'valence':[0.0,1.0], 
                   'tempo':[0,None],
                   'duration_ms':[0,None],
                    'time_signature':[0,None] }
descriptions = {'danceability':'Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.', 
                   'energy':'Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.', 
                   'loudness':'The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.', 
                   'key':'The key the track is in. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.',
                   'mode':'Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.', 
                   'speechiness':'Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.',
                   'acousticness':'A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.', 
                   'instrumentalness':'Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.', 
                   'liveness':'Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.', 
                   'valence':'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).', 
                   'tempo':'The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.',
                   'duration_ms':'The duration of the track in milliseconds.',
                    'time_signature':'An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).' }


def description(prop):
    '''
    Input: property
    Output description
    '''
    return descriptions[prop]
    

def manh_distance(df, index1, index2, columns):
    '''
    Input: dataframe, index1 & 2, columns with properties
    Output: mahnhattan distance between indexes 1 & 2
    '''
    summ = 0
    for item in columns:
        summ += abs(df.loc[index1,item] - df.loc[index2,item])
    return summ

def eucl_distance(df, index1, index2, columns):
    '''
    Input: dataframe, index1 & 2, columns with properties
    Output: euclidian distance between indexes 1 & 2
    '''
    summ = 0
    for item in columns:
        summ += (df.loc[index1,item] - df.loc[index2,item])**2
    return summ**0.5

def distance_to_centroid(df,centroid_pos):
    centroid_distances = []
    # iterate through rows of data frame
    for index in range(df.shape[0]):
        summ = 0
        # iterate through properties
        for col_num in range(df.shape[1]):
            summ += (df.iloc[index,col_num] - centroid_pos[col_num])**2
        distance = summ**0.5
        centroid_distances.append(distance)
    return centroid_distances

def get_distance_matrix(df, columns, dist_type, output ='list'):
    '''
    Input:  dataframe,
            list of columns with properties
            distance type: 'euclid' or 'manhattan'
            output: list, dataframe or graph of distance matrix
    Output: list, dataframe or graph of distance matrix
    Uses: manh_distance(), eucl_distance()
    '''
    # get list of indexes
    indexes = df.index
    distances = []
    # iterate through indexes
    for ind in indexes:
        col = []
        # iterate through columns
        for item in indexes:
            # what distance to calculate?
            if dist_type == 'euclid':
                difference = eucl_distance(df,ind,item,columns)
            elif dist_type == 'manhattan':
                difference = manh_distance(df,ind,item,columns)
            col.append(difference)
        distances.append(col)
    #  what to return ?
    if output == 'graph':
        df_distances = pd.DataFrame(distances)
        df_distances.index = df.index
        df_distances.columns = df.index
        sns.heatmap(df_distances)
    elif output == 'list':
        return distances
    elif output == 'dataframe':
        df_distances = pd.DataFrame(distances)
        df_distances.index = df.index
        df_distances.columns = df.index
        return df_distances
    
def normalise_ratings(col,min_value,max_value): 
    
    '''
    Input: column values, min & max values
    Output: list of normalized values
    Note: add different normalizasions
    '''
    norm_col = []
    for value in col: 
        norm_col.append(
            (value - min_value) / (max_value - min_value)
        ) 
    return norm_col

def get_random_song(df):
    '''
    Input: df - spotify song list
    Output: random song tuple - (name,artist)
    '''
    len_song_list = df.shape[0]
    song_number = random.randint(0,len_song_list)
    song_name_and_artist = df.iloc[song_number,:].name
    return song_name_and_artist

def get_random_song_cluster(df, cluster, num_songs):
    '''
    Input: df - spotify song list with clusters
    Output: list of random songs
    '''
    songs_random = []
    songs_cluster = df[df['cluster'] == cluster]
    for i in range(num_songs):
        songs_random.append(get_random_song(songs_cluster))
    return songs_random
        