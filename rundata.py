import os,pandas,time
from dbutils import RunData,printlist
def explore_rundata(num=5):
    files = os.listdir(RunData.datadir)
    paths = list()
    ignored = list()
    for f in files:
        p = os.path.join(RunData.datadir,f)
        try:
            paths.append((p,int(os.path.basename(f).strip('.pkl'))))
        except:
            ignored.append(p)
    if(len(ignored) > 0):
        print("ignored:")
        printlist(ignored)
    paths.sort(key=lambda x: x[1],reverse=True)
    if not num:
        num = len(paths) - 1
    for i,p in enumerate(paths[:num],1):
        print(str(i)+".",time.ctime(p[1]))
    choice = input("choose data to load: ")
    data = pandas.read_pickle(paths[int(choice) - 1][0])
    data.when()
    data.how()
    print("\nprint data.show() to view the results of this run")
    print("print x = explore_rundata(): to load another run as x")
    return data

if __name__ == '__main__':
    data = explore_rundata()
