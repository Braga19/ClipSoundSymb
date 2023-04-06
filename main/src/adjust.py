def subset(df, name, rating, attribute, value):
  '''take a pandas df and output a subset of the df where value is gender, age or polarity'''
  return df.loc[df[attribute] == value, [name, rating]]