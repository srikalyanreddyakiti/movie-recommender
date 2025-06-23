# Movie Recommendation Sytem leveraging  Artification Intelligence Techniques

## Project Summary 
This project deals with the recommendation of the movies recommendation based on the inputs from the user.It learns from the user's previous inputs and gives better and more refined recommendations over time.

## Functionalities
- Recommend movies which are similar to the genre of the choice selected by the user.
- A wide list of 24 genres viz. action,comedy,adventure,romance,etc that are available to the user to select.

## Technologies
We use the below technologies to implement our project.
- Python (for the server side processing)
- Machine Learning algorithms (for training the model)
- HTML and Bootstrap (for the fronend)
- (optional) Visual Studio Code (IDE for running the code)
  
## Installation
- We must make sure that the latest version of Python is installed in our system .
- We have to make sure that Anaconda is also installed in order to run the Jupyter source file especially in the IDE.

## Project Structure
This is the structure of the project.

- `app.py ` : This is the file from the which the program 
  is run.
- `README.md` : The file which contains the documentation for our project.  
- templates\
    - `home.html` : The home page where the user interface for the project is present.
    - `movies.html` : The page where the recommended movies based on the user choice are displayed.
- `MRS.ipynb` : The Jupyter source file which contains the Machine Learning logic for training the models.
- `movies_with_ratings.pkl` : The pickle file containing the saved model which was trained for merging the movie data with the ratings.
- `vectorizer.pkl` :  The pickle file containing the saved model which was trained for vectorizing the movie ratings.
  
## Execution
 We have open the terminal in the path which contains our code folder and run the following command to start our program.

 ```bash
 python app.py 
 ``` 

## Datasets 
 We have 2 datasets on which we train our models. They are:
 - `movies.csv` : This file contains all the information regarding the movies like Title,Genre and the year of release.
 - `ratings.csv` : This file contains the Movie Id, rating and timestamp(The exact time at which the rating was given)

## Contributions
 In order to contribute to our project, you can fork the repository from Github

## Licenses 
This project is a university academic assignment and is not intended for commercial purposes. All rights reserved. Redistribution or reuse is prohibited unless permitted under the  university's policies.

