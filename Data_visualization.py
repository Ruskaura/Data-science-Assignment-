
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Visualizing Quantities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Name</th>\n",
       "      <th>ScientificName</th>\n",
       "      <th>Category</th>\n",
       "      <th>Order</th>\n",
       "      <th>Family</th>\n",
       "      <th>Genus</th>\n",
       "      <th>ConservationStatus</th>\n",
       "      <th>MinLength</th>\n",
       "      <th>MaxLength</th>\n",
       "      <th>MinBodyMass</th>\n",
       "      <th>MaxBodyMass</th>\n",
       "      <th>MinWingspan</th>\n",
       "      <th>MaxWingspan</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Black-bellied whistling-duck</td>\n",
       "      <td>Dendrocygna autumnalis</td>\n",
       "      <td>Ducks/Geese/Waterfowl</td>\n",
       "      <td>Anseriformes</td>\n",
       "      <td>Anatidae</td>\n",
       "      <td>Dendrocygna</td>\n",
       "      <td>LC</td>\n",
       "      <td>47.0</td>\n",
       "      <td>56.0</td>\n",
       "      <td>652.0</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>76.0</td>\n",
       "      <td>94.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Fulvous whistling-duck</td>\n",
       "      <td>Dendrocygna bicolor</td>\n",
       "      <td>Ducks/Geese/Waterfowl</td>\n",
       "      <td>Anseriformes</td>\n",
       "      <td>Anatidae</td>\n",
       "      <td>Dendrocygna</td>\n",
       "      <td>LC</td>\n",
       "      <td>45.0</td>\n",
       "      <td>53.0</td>\n",
       "      <td>712.0</td>\n",
       "      <td>1050.0</td>\n",
       "      <td>85.0</td>\n",
       "      <td>93.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Snow goose</td>\n",
       "      <td>Anser caerulescens</td>\n",
       "      <td>Ducks/Geese/Waterfowl</td>\n",
       "      <td>Anseriformes</td>\n",
       "      <td>Anatidae</td>\n",
       "      <td>Anser</td>\n",
       "      <td>LC</td>\n",
       "      <td>64.0</td>\n",
       "      <td>79.0</td>\n",
       "      <td>2050.0</td>\n",
       "      <td>4050.0</td>\n",
       "      <td>135.0</td>\n",
       "      <td>165.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Ross's goose</td>\n",
       "      <td>Anser rossii</td>\n",
       "      <td>Ducks/Geese/Waterfowl</td>\n",
       "      <td>Anseriformes</td>\n",
       "      <td>Anatidae</td>\n",
       "      <td>Anser</td>\n",
       "      <td>LC</td>\n",
       "      <td>57.3</td>\n",
       "      <td>64.0</td>\n",
       "      <td>1066.0</td>\n",
       "      <td>1567.0</td>\n",
       "      <td>113.0</td>\n",
       "      <td>116.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Greater white-fronted goose</td>\n",
       "      <td>Anser albifrons</td>\n",
       "      <td>Ducks/Geese/Waterfowl</td>\n",
       "      <td>Anseriformes</td>\n",
       "      <td>Anatidae</td>\n",
       "      <td>Anser</td>\n",
       "      <td>LC</td>\n",
       "      <td>64.0</td>\n",
       "      <td>81.0</td>\n",
       "      <td>1930.0</td>\n",
       "      <td>3310.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>165.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           Name          ScientificName  \\\n",
       "0  Black-bellied whistling-duck  Dendrocygna autumnalis   \n",
       "1        Fulvous whistling-duck     Dendrocygna bicolor   \n",
       "2                    Snow goose      Anser caerulescens   \n",
       "3                  Ross's goose            Anser rossii   \n",
       "4   Greater white-fronted goose         Anser albifrons   \n",
       "\n",
       "                Category         Order    Family        Genus  \\\n",
       "0  Ducks/Geese/Waterfowl  Anseriformes  Anatidae  Dendrocygna   \n",
       "1  Ducks/Geese/Waterfowl  Anseriformes  Anatidae  Dendrocygna   \n",
       "2  Ducks/Geese/Waterfowl  Anseriformes  Anatidae        Anser   \n",
       "3  Ducks/Geese/Waterfowl  Anseriformes  Anatidae        Anser   \n",
       "4  Ducks/Geese/Waterfowl  Anseriformes  Anatidae        Anser   \n",
       "\n",
       "  ConservationStatus  MinLength  MaxLength  MinBodyMass  MaxBodyMass  \\\n",
       "0                 LC       47.0       56.0        652.0       1020.0   \n",
       "1                 LC       45.0       53.0        712.0       1050.0   \n",
       "2                 LC       64.0       79.0       2050.0       4050.0   \n",
       "3                 LC       57.3       64.0       1066.0       1567.0   \n",
       "4                 LC       64.0       81.0       1930.0       3310.0   \n",
       "\n",
       "   MinWingspan  MaxWingspan  \n",
       "0         76.0         94.0  \n",
       "1         85.0         93.0  \n",
       "2        135.0        165.0  \n",
       "3        113.0        116.0  \n",
       "4        130.0        165.0  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Read the data and display the first few rows\n",
    "birds = pd.read_csv(\"../data/birds.csv\")\n",
    "birds.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(443, 13)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the dimension of the dataset\n",
    "birds.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 443 entries, 0 to 442\n",
      "Data columns (total 13 columns):\n",
      " #   Column              Non-Null Count  Dtype  \n",
      "---  ------              --------------  -----  \n",
      " 0   Name                443 non-null    object \n",
      " 1   ScientificName      443 non-null    object \n",
      " 2   Category            443 non-null    object \n",
      " 3   Order               443 non-null    object \n",
      " 4   Family              443 non-null    object \n",
      " 5   Genus               443 non-null    object \n",
      " 6   ConservationStatus  443 non-null    object \n",
      " 7   MinLength           443 non-null    float64\n",
      " 8   MaxLength           443 non-null    float64\n",
      " 9   MinBodyMass         443 non-null    float64\n",
      " 10  MaxBodyMass         443 non-null    float64\n",
      " 11  MinWingspan         443 non-null    float64\n",
      " 12  MaxWingspan         443 non-null    float64\n",
      "dtypes: float64(6), object(7)\n",
      "memory usage: 45.1+ KB\n"
     ]
    }
   ],
   "source": [
    "birds.info()"
   ]
  },
  
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "443"
      ]
     },
     
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "birds.Name.nunique()"
   ]
  },

  
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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DXJzczF69Gj84x//wBtvvIGpU6d2wVMmIjlhEy8RBeMn7QMjZ9wHhkgZfrZ0PcpMrQC4DwwRnad9YIiIiIikwABDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0RERIrDAENERESKwwBDREREisMAQ0SSEqQeABEpEgMMERERKQ4DDBFJSiX1AIhIkRhgiIiISHEYYIiIiEhxGGCIiIhIcRhgiIiISHEYYIiIiEhxGGCIiIhIcRhgiIiISHEYYIiIiEhxGGCISFI8lQARBYMBhoiIiBSHAYaIJMVTCRBRMBhgiIiISHEYYIiIiEhxGGCISFJs4iWiYDDAEBERkeIwwBCRpNjES0TBYIAhIiIixWGAISIiIsVhgCEiSbGJl4iCwQBDREREisMAQ0RERIrDAENEkuIqJCIKBgMMERERKQ4DDBFJik28RBQMBhgiIiJSHAYYIiIiUhwGGCKSFJt4iSgYDDBERESkOAwwRCQpNvESUTAYYIiIiEhxGGCIiIhIcRhgiIiISHEYYIhIUlyFRETBYIAhIkmxiZeIgsEAQ0RERIrDAENERESKwwBDRLIhCJxQIqLAdDrAbNq0Cddffz3S0tKgUqnw2WefeVz/61//GiqVyuNj2rRpHsfU1tZi7ty50Ov1iI2Nxfz589HY2OhxzJ49e3DZZZchLCwM6enpeOaZZzr/7IhI9tybeJlfiChQnQ4wTU1NGD16NJYtW3bGY6ZNm4by8nLx48MPP/S4fu7cuSgqKkJubi5Wr16NTZs24a677hKvN5vNmDJlCjIyMlBQUIBnn30Wjz/+OP797393drhEpCDML0QUqJDO3mD69OmYPn36WY/R6XQwGAx+rztw4ADWrl2LHTt2YNy4cQCAV155Bddeey2ee+45pKWl4f3330dbWxvefPNNaLVaDB8+HIWFhXj++ec9gg4RKR9DCxEFo1t6YDZu3Ijk5GQMHjwY99xzD2pqasTr8vPzERsbK4YXAMjOzoZarca2bdvEYy6//HJotVrxmKlTp6K4uBh1dXV+H9NiscBsNnt8EJGysAeG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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "birds.plot(kind=\"line\")"
   ]
  },
  
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
