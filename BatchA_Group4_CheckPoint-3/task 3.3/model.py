import numpy as np
import pandas as pd
import pickle

bookm = pd.read_csv("BOOKSMASTERTRAIN.csv")
user = pd.read_csv("USERMASTER.csv")
bookcat=pd.read_csv("BOOKSCATALOGUE.csv")
bookph = pd.read_csv("BOOKSPURCHHISTORY.csv")
bookvh =  pd.read_csv("BOOKSVISITHISTORY.csv")
bookss = pd.read_csv("SAMPLESUBMISSION.csv")

bk=bookm.copy()
bk.sort_values(by='USERRATINGS',ascending=False,inplace=True)
bk['SERIES'].fillna(" ",inplace=True)
bk['BOOK']= bk['BOOKNAME']+bk['SERIES']

bookph['TIMESTAMP'] =pd.to_datetime(bookph['TIMESTAMP'])
bookph.sort_values(by="TIMESTAMP",ascending=False,inplace=True)
df = bookph[['UserID','BookID']]
d= df.groupby("UserID").groups
s = pd.DataFrame({"USERID":d.keys()})
s['INDEX']= [list(d[x]) for x in d.keys()]

p=[]
for x in s["INDEX"].values:
    k=list(map(lambda y:bookph.loc[y,"BookID"],x))
    p.append(k)
s['BOOKS']=p
s.drop('INDEX',axis=1,inplace=True)

sample=pd.merge(bookss,s,on="USERID",how='left')
sample.drop('PURCHASEDBOOKID',axis=1,inplace=True)
sample["BOOKS"].fillna("",inplace=True)

bookm =bookm.get(['BookID','GENRE','BOOKNAME','SERIES','USERRATINGS'])
bookm["SERIES"].fillna("",inplace=True)

bookm["Book"] = bookm["BOOKNAME"]+bookm["SERIES"]
bookm.drop(columns=['BOOKNAME','SERIES'],inplace=True)

bookph = bookph.get(['UserID','BookID'])
bookph.drop_duplicates(inplace=True)

bookpur = pd.DataFrame(bookph.groupby('UserID')['BookID'].count().reset_index())
bookpur.rename(columns={"BookID":"No_Of_Books_Purchased"},inplace=True)

bookpur= bookpur[bookpur["No_Of_Books_Purchased"]>=3]
x= bookpur["UserID"]
bookph = bookph[bookph["UserID"].isin(x)]

da = pd.merge(bookph,user,on="UserID",how ="left")
df=da.get(['UserID','BookID','AGEGROUP','GENDER'])

fin = pd.merge(df,bookm,on = "BookID",how = "left")
fin.dropna(subset= ['Book'],inplace=True)

ratings = fin.groupby('BookID')['USERRATINGS'].count()
rat = pd.DataFrame(ratings.reset_index())
rat.rename(columns={'USERRATINGS':"No_Of_Userratings"},inplace=True)

rat = rat[rat["No_Of_Userratings"]>5]

findata= pd.merge(fin,rat,on="BookID",how="right")

findata.sort_values(by=["USERRATINGS",'No_Of_Userratings'],ascending=False,inplace=True)

j=0
books=[]
for i in findata['Book'].unique():
    if j<10:
        books.append(i)
    j+=1

book_pivot = findata.pivot(columns='UserID',index="BookID",values="USERRATINGS")
book_pivot.fillna(0,inplace=True)

from scipy.sparse import csr_matrix
book_pred= csr_matrix(book_pivot)

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(algorithm = 'brute')
model_knn.fit(book_pred)

def book(user):
    y=0
    user2 =sample[sample['USERID']==user]
    for x in user2.BOOKS.values:
        if len(x)==0:
            return books
        else:
            for i in x:
                if i in findata['BookID'].values:
                    y=i
                    break
            if y==0:
                return books
            else:
                return y

def bookname(bookid):
    bk = bookm[bookm["BookID"]==bookid]
    for x in bk.Book.values:
        return x

pickle.dump(model_knn, open('model.pkl','wb'))