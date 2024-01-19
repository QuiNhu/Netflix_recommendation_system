from flask import Flask, render_template, request
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import random

nrows = 100000
#load movie mapping file
df_title = pd.read_csv('data_Netflix/movie_titles_v2.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], nrows=4155).iloc[:,:3]
df_title.set_index('Movie_Id', inplace = True)
df_title['image'] = df_title['Name'].str.lower()
characters_to_remove = [":", "'", "#", "/", "?", "!", "%", "$", "*", ".", "&"]
for char in characters_to_remove:
    df_title['image'] = df_title['image'].str.replace(char, "")

df_title['image'] = df_title['image'].str.replace(" ", "_")
#df_title['image'] = df_title['image'].str.replace(r"[::'#/?!%*.$&]", "")
df_title['image'] = df_title['image'].str.cat(['.jpg'] * len(df_title))
df_title.to_csv('data_Netflix/movie_titles.csv')
movie_images = dict(zip(df_title['Name'], df_title['image']))

df = pd.read_csv('data_Netflix/cleaned_data.csv', nrows=nrows).iloc[:,1:]
df_movie_summary = pd.read_csv('data_Netflix/movie_summary.csv')
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

df_cust_summary = pd.read_csv('data_Netflix/cus_movi.csv')
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

df_p = pd.read_csv('data_Netflix/data_pivot.csv')

### Recommend with Collaborative Filtering
reader = Reader()
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']].iloc[:nrows], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
svd.fit(trainset)
print('Model complete!!')

user_list = df.Cust_Id.astype(int).to_list()
movie_title_list = ['8 Man', 'My Favorite Brunette', 'Lord of the Rings: The Return of the King: Extended Edition: Bonus Material', 
                    'Nature: Antarctica', 'Immortal Beloved', "By Dawn's Early Light", 'Strange Relations', 'Chump Change', 
                    'My Bloody Valentine', 'Inspector Morse 31: Death Is Now My Neighbour', 'Never Die Alone', 
                    "Sesame Street: Elmo's World: The Street We Live On", 'Lilo and Stitch', "Something's Gotta Give", 
                    'Classic Albums: Meat Loaf: Bat Out of Hell', "ABC Primetime: Mel Gibson's The Passion of the Christ", 'Spitfire Grill', 
                    'The Bad and the Beautiful', 'A Yank in the R.A.F.', 'The Weather Underground', 'Jade', 'Outside the Law', 
                    'Barbarian Queen 2', 'WWE: Armageddon 2003', 'Jingle All the Way', 'The Powerpuff Girls Movie', 'Elfen Lied', 'Iron Monkey 2', 
                    'Record of Lodoss War: Chronicles of the Heroic Knight', 'They Came Back', 'A Fishy Story', 'Sam the Iron Bridge', 'Scandal', 
                    "Bruce Lee: A Warrior's Journey", 'Bear Cub', 'Silk Stockings', 'Travel the World by Train: Africa', 'Plain Dirty', 
                    'Beyonce: Live at Wembley', 'Drowning on Dry Land', 'Lost in the Wild', 'Goddess of Mercy', 'Get Out Your Handkerchiefs']
input_list = random.sample(user_list, 60) + movie_title_list

def predict_movies_for_user(user_id, df_title, drop_movie_list, svd, top_n=12):
    user = df_title.copy().reset_index()
    user = user[~user['Movie_Id'].isin(drop_movie_list)]
    user['Estimate_Score'] = user['Movie_Id'].apply(lambda x: svd.predict(user_id, x).est)
    user = user.drop('Movie_Id', axis=1)

    user = user.sort_values('Estimate_Score', ascending=False)
    user['Estimate_Score'] = round(user['Estimate_Score'], 2)
    recommended_movies = user.head(top_n)

    return recommended_movies

### Recommend with Pearsons' correlations ## df_title, df_p, df_movie_summary
def recommend(movie_title):
    print("For movie ({})".format(movie_title))
    if len(df_title['Name'].str.lower() == movie_title.lower()) != 0:
      print("- Top 20 movies recommended based on Pearsons' correlation - ")
      i = int(df_title.index[df_title['Name'].str.lower() == movie_title.lower()][0])
      target = df_p[str(i)]
      similar_to_target = df_p.corrwith(target)
      corr_target = pd.DataFrame(similar_to_target, columns = ['Estimate_Score'])
      corr_target.dropna(inplace = True)
      corr_target = corr_target.sort_values('Estimate_Score', ascending = False)
      corr_target['Estimate_Score'] = round(corr_target['Estimate_Score'], 2)
      corr_target = corr_target.drop('Cust_Id')
      corr_target.index = corr_target.index.map(int)
      corr_target = corr_target.join(df_title).join(df_movie_summary)[['Year', 'Name', 'image', 'Estimate_Score']]
      return corr_target[:21]#.to_string(index=False)
    else:
      return f"We can't the {movie_title} film!!! SORRY!!"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', input_list=input_list)

@app.route('/result', methods=['POST'])
def result():
    input_value = request.form['input_value']

    try:
        input_value = int(input_value)
        recommended_movies = predict_movies_for_user(input_value, df_title, drop_movie_list, svd)
        #print(recommended_movies)
        recommendation_data = recommended_movies.to_dict('records')
        return render_template('result.html', recommendation=recommendation_data)
    except ValueError:
        try:
            recommended_movies = recommend(input_value)
            #print(recommended_movies)
            recommendation_data = recommended_movies.to_dict('records')
            return render_template('result.html', recommendation=recommendation_data)
        except KeyError:
            return f"We can't find the movie for input: {input_value}!!! SORRY!!"

if __name__ == '__main__':
    app.run(debug=True)