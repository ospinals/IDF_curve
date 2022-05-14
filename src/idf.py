def ano(ts):
    return ts.year


def plotdata(df,titulo):
    '''Toma un DataFrame y grafica la infromación de precipitacion y le pone el titulo'''
    plt.figure(figsize=(14,5))
    plt.plot(df.fecha,df.P,'-',color='Darkblue')
    plt.title(titulo)
    plt.xlabel('Fecha')
    plt.ylabel('Precipitacion [mm]')
    plt.grid()
    
def mes(ts):
    return ts.month
    
def cicloanual(df):
    '''Toma un DataFrame y grafica el ciclo anual de precipitacion'''
    plt.figure(figsize=(14,5))
    dk=df.resample('M',on='fecha').sum()
    dk['meses']=pd.Series(dk.index).apply(mes).values
    M=dk.groupby('meses').mean().P
    plt.plot(range(12),M,color='Darkblue')
    plt.title('Ciclo anual')
    plt.xlabel('Mes')
    plt.ylabel('Precipitacion [mm]')
    plt.xticks(range(12),['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
    plt.grid()
    plt.ylim(0,300)
    return M

def resolucion(df,t):
    '''Cambia la resolucion de los datos de un DataFrame segun el valor deseado t en minutos'''
    df=df.set_index(df.fecha).groupby(pd.TimeGrouper(freq='{}min'.format(t))).sum()
    df['fecha']=df.index
    return df

def acumulados(df,dur,res):
    '''Calcula acumulados de precipitacion del DataFrame por medio de la suma movil\
    dur:duraciones en minutos [array]
    res:resolucion temporal de los datos [minutos]
    '''
    paso=int(dur)/int(res)
#     df['acumulados']=pd.Series([sum(df.P[i:i+paso]) for i in range(len(df.P))]).values
#     print df.acumulados
    df['acumulados']=df.P.rolling(paso).sum()
    df['year']=df.fecha.apply(ano)
    df=df.fillna(0)
    df.acumulados[df.acumulados<0]=0
    return df.groupby('year').max()

def maxi(df,dur,res):
    '''Calcula la matriz de acumulados maximos anuales para un DataFrame dado un array con duraciones y \
    la resolucion temporal de los datos'''
    maximos=pd.DataFrame()
    maximos['year']=pd.Series(list(set(df.fecha.apply(ano))))
    for i in dur:
        maximos[str(i)]=acumulados(df,i,res).acumulados.values
        
    return maximos

def inten(df):
    '''Convierte la matriz de acumualdos de precipitacion en intensidades horarias [mm/hora]'''
    for i in df.keys()[1:]:
        df[i]=df[i]/(float(i)/60.)
    return df

def normfit(intensidad):
    param=sp.norm.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.norm.pdf(x,loc=param[0],scale=param[1])
    plt.plot(x,curva,label='Normal')
    return sp.kstest(intensidad,'norm',param)[1]
    
def lognormal(intensidad):
    param=sp.lognorm.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.lognorm.pdf(x,s=param[0],loc=param[1],scale=param[2])
    plt.plot(x,curva,label='Log-Normal')
    return sp.kstest(intensidad,'lognorm',param)[1]
  

def gumbel(intensidad):
    param=sp.gumbel_r.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.gumbel_r.pdf(x,loc=param[0],scale=param[1])
    plt.plot(x,curva,label='Gumbel')
    return sp.kstest(intensidad,'gumbel_r',param)[1]
    
    
def pearson(intensidad):
    param=sp.pearson3.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.pearson3.pdf(x,skew=param[0],loc=param[1],scale=param[2])
    plt.plot(x,curva,label='Pearson3')
    return sp.kstest(intensidad,'pearson3',param)[1]

    
def weibull(intensidad):
    param=sp.weibull_max.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.weibull_max.pdf(x,c=param[0],loc=param[1],scale=param[2])
    plt.plot(x,curva,label='Weibull')
    return sp.kstest(intensidad,'weibull_max',param)[1]

    
def histintensidad(df,i):
    '''Grafica dada una duracion [horas] dado un DataFrame que contenga la matriz de intensidad'''
    i=str(i)
    plt.figure()
    dur=[float(i)/60.]*len(df[i])
    intensidad=df[i]
    
    nfit=normfit(intensidad)
    lfit=lognormal(intensidad)
    gfit=gumbel(intensidad)
    pfit=pearson(intensidad)
    wfit=weibull(intensidad)
    hist,bins=np.histogram(intensidad,bins=8)
    hist=hist/np.sum(hist.astype(float))
    plt.plot(bins[:-1],hist,'-o',label='{} horas'.format(str(float(i)/60.)),color='k')
    plt.xlim(0)
    
    plt.ylim([0,1])
    plt.grid(linestyle='--')
    plt.legend()
    plt.xlabel('Intensidad [mm/hora]')
    plt.ylabel('Frecuencia')
    plt.title('Intensidades')
    
    dic={}
    dic['Normal']=nfit
    dic['Log-Normal']=lfit
    dic['Gumbel']=gfit
    dic['Pearson3']=pfit
    dic['Weibull']=wfit
    
    return dic

def rename(k,periodos):
    kl=k.transpose()
    g={}
    for w,t in zip(kl.keys(),periodos):
        g[w]=t
        
    return kl.rename(columns=g)
    
def gumbelT(intensidad,q):
    param=sp.gumbel_r.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    return sp.gumbel_r.ppf(q,loc=param[0],scale=param[1])

def normalT(intensidad,q):
    param=sp.normal.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    return sp.normal.ppf(q,loc=param[0],scale=param[1])

def prob(t):
    return 1.-(1./t)

def idf(df,dura,T):
    resultados=pd.DataFrame()
    
    for i in dura:
        intensidad=df[str(i)]
        resultados[i]=gumbelT(intensidad,prob(T))
    
    resultados=rename(resultados,T)     
    resultados=resultados.set_index(resultados.index/60.)
    plt.figure()
    resultados.plot()
    
    plt.xlabel('Duracion [horas]')
    plt.ylabel('Intensidad [mm/hora]')
    plt.title('Curvas IDF')
    plt.grid(linestyle='--')
    plt.savefig('IDF.png',dpi=300,bbox_inches='tight')
    return resultados.round(2)

def estadisticas(dfi,dur):
    dicti={}
    for i in dur:
        dicti[i]=histintensidad(dfi,i)
    a=pd.DataFrame(dicti).transpose()
    plt.figure()
    a.boxplot(showfliers=True)
    plt.ylabel('Bondad de ajuste')
    return a

def IDF(df,res,duraciones=np.arange(15,1455,15),periodos=[2.33,5,10,25,50,100]):
    periodos=np.array(periodos)
    duraciones=np.array(duraciones)
    df=resolucion(df,res)
    dfmax=maxi(df,duraciones,res)
    try:
        dfmax=dfmax.drop((np.where(np.array(Geronamax)==0)[0][0])).reset_index(drop=True)
    except:
        pass
    dfInt=inten(dfmax)
    g=estadisticas(dfInt,duraciones)
    gh=idf(dfInt,duraciones,periodos)
#     Text=np.log(np.array([Tr for i in range(len(duraciones))]).T.reshape(len(duraciones)*len(Tr)))
#     Dext=np.log(np.array([duraciones for i in range(len(Tr))]).reshape(len(duraciones)*len(Tr)))
#     Ones=np.ones(Dext.shape)
#     X=np.vstack((Ones,Dext,Text)).T
#     Y=np.log(np.reshape(IntAdjusted,(IntAdjusted.size)))
#     param,residual=np.linalg.lstsq(X,Y)[:2]
#     k=np.exp(param[0]); n=-1*param[1]; m=param[2]
#     R2 = 1 - residual / (Y.size * Y.var())
    return gh




def ano(ts):
    return ts.year


def plotdata(df,titulo):
    '''Toma un DataFrame y grafica la infromación de precipitacion y le pone el titulo'''
    plt.figure(figsize=(14,5))
    plt.plot(df.fecha,df.P,'-',color='Darkblue')
    plt.title(titulo)
    plt.xlabel('Fecha')
    plt.ylabel('Precipitacion [mm]')
    plt.grid()
    
def mes(ts):
    return ts.month
    
def cicloanual(df):
    '''Toma un DataFrame y grafica el ciclo anual de precipitacion'''
    plt.figure(figsize=(14,5))
    dk=df.resample('M',on='fecha').sum()
    dk['meses']=pd.Series(dk.index).apply(mes).values
    M=dk.groupby('meses').mean().P
    plt.plot(range(12),M,color='Darkblue')
    plt.title('Ciclo anual')
    plt.xlabel('Mes')
    plt.ylabel('Precipitacion [mm]')
    plt.xticks(range(12),['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
    plt.grid()
    plt.ylim(0,300)
    return M

def resolucion(df,t):
    '''Cambia la resolucion de los datos de un DataFrame segun el valor deseado t en minutos'''
    df=df.set_index(df.fecha).groupby(pd.TimeGrouper(freq='{}min'.format(t))).sum()
    df['fecha']=df.index
    return df

def acumulados(df,dur,res):
    '''Calcula acumulados de precipitacion del DataFrame por medio de la suma movil\
    dur:duraciones en minutos [array]
    res:resolucion temporal de los datos [minutos]
    '''
    paso=int(dur)/int(res)
#     df['acumulados']=pd.Series([sum(df.P[i:i+paso]) for i in range(len(df.P))]).values
#     print df.acumulados
    df['acumulados']=df.P.rolling(paso).sum()
    df['year']=df.fecha.apply(ano)
    df=df.fillna(0)
    df.acumulados[df.acumulados<0]=0
    return df.groupby('year').max()

def maxi(df,dur,res):
    '''Calcula la matriz de acumulados maximos anuales para un DataFrame dado un array con duraciones y \
    la resolucion temporal de los datos'''
    maximos=pd.DataFrame()
    maximos['year']=pd.Series(list(set(df.fecha.apply(ano))))
    for i in dur:
        maximos[str(i)]=acumulados(df,i,res).acumulados.values
        
    return maximos

def inten(df):
    '''Convierte la matriz de acumualdos de precipitacion en intensidades horarias [mm/hora]'''
    for i in df.keys()[1:]:
        df[i]=df[i]/(float(i)/60.)
    return df

def normfit(intensidad):
    param=sp.norm.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.norm.pdf(x,loc=param[0],scale=param[1])
    plt.plot(x,curva,label='Normal')
    return sp.kstest(intensidad,'norm',param)[1]
    
def lognormal(intensidad):
    param=sp.lognorm.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.lognorm.pdf(x,s=param[0],loc=param[1],scale=param[2])
    plt.plot(x,curva,label='Log-Normal')
    return sp.kstest(intensidad,'lognorm',param)[1]
  

def gumbel(intensidad):
    param=sp.gumbel_r.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.gumbel_r.pdf(x,loc=param[0],scale=param[1])
    plt.plot(x,curva,label='Gumbel')
    return sp.kstest(intensidad,'gumbel_r',param)[1]
    
    
def pearson(intensidad):
    param=sp.pearson3.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.pearson3.pdf(x,skew=param[0],loc=param[1],scale=param[2])
    plt.plot(x,curva,label='Pearson3')
    return sp.kstest(intensidad,'pearson3',param)[1]

    
def weibull(intensidad):
    param=sp.weibull_max.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    curva=sp.weibull_max.pdf(x,c=param[0],loc=param[1],scale=param[2])
    plt.plot(x,curva,label='Weibull')
    return sp.kstest(intensidad,'weibull_max',param)[1]

    
def histintensidad(df,i):
    '''Grafica dada una duracion [horas] dado un DataFrame que contenga la matriz de intensidad'''
    i=str(i)
    plt.figure()
    dur=[float(i)/60.]*len(df[i])
    intensidad=df[i]
    
    nfit=normfit(intensidad)
    lfit=lognormal(intensidad)
    gfit=gumbel(intensidad)
    pfit=pearson(intensidad)
    wfit=weibull(intensidad)
    hist,bins=np.histogram(intensidad,bins=8)
    hist=hist/np.sum(hist.astype(float))
    plt.plot(bins[:-1],hist,'-o',label='{} horas'.format(str(float(i)/60.)),color='k')
    plt.xlim(0)
    
    plt.ylim([0,1])
    plt.grid(linestyle='--')
    plt.legend()
    plt.xlabel('Intensidad [mm/hora]')
    plt.ylabel('Frecuencia')
    plt.title('Intensidades')
    
    dic={}
    dic['Normal']=nfit
    dic['Log-Normal']=lfit
    dic['Gumbel']=gfit
    dic['Pearson3']=pfit
    dic['Weibull']=wfit
    
    return dic

def rename(k,periodos):
    kl=k.transpose()
    g={}
    for w,t in zip(kl.keys(),periodos):
        g[w]=t
        
    return kl.rename(columns=g)
    
def gumbelT(intensidad,q):
    param=sp.gumbel_r.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    return sp.gumbel_r.ppf(q,loc=param[0],scale=param[1])

def normalT(intensidad,q):
    param=sp.normal.fit(intensidad)
    x=np.linspace(np.min(intensidad),np.max(intensidad),100)
    return sp.normal.ppf(q,loc=param[0],scale=param[1])

def prob(t):
    return 1.-(1./t)

def idf(df,dura,T):
    resultados=pd.DataFrame()
    
    for i in dura:
        intensidad=df[str(i)]
        resultados[i]=gumbelT(intensidad,prob(T))
    
    resultados=rename(resultados,T)     
    resultados=resultados.set_index(resultados.index/60.)
    plt.figure()
    resultados.plot()
    
    plt.xlabel('Duracion [horas]')
    plt.ylabel('Intensidad [mm/hora]')
    plt.title('Curvas IDF')
    plt.grid(linestyle='--')
    plt.savefig('IDF.png',dpi=300,bbox_inches='tight')
    return resultados.round(2)

def estadisticas(dfi,dur):
    dicti={}
    for i in dur:
        dicti[i]=histintensidad(dfi,i)
    a=pd.DataFrame(dicti).transpose()
    plt.figure()
    a.boxplot(showfliers=True)
    plt.ylabel('Bondad de ajuste')
    return a

def IDF(df,res,duraciones=np.arange(15,1455,15),periodos=[2.33,5,10,25,50,100]):
    periodos=np.array(periodos)
    duraciones=np.array(duraciones)
    df=resolucion(df,res)
    dfmax=maxi(df,duraciones,res)
    try:
        dfmax=dfmax.drop((np.where(np.array(Geronamax)==0)[0][0])).reset_index(drop=True)
    except:
        pass
    dfInt=inten(dfmax)
    g=estadisticas(dfInt,duraciones)
    gh=idf(dfInt,duraciones,periodos)
#     Text=np.log(np.array([Tr for i in range(len(duraciones))]).T.reshape(len(duraciones)*len(Tr)))
#     Dext=np.log(np.array([duraciones for i in range(len(Tr))]).reshape(len(duraciones)*len(Tr)))
#     Ones=np.ones(Dext.shape)
#     X=np.vstack((Ones,Dext,Text)).T
#     Y=np.log(np.reshape(IntAdjusted,(IntAdjusted.size)))
#     param,residual=np.linalg.lstsq(X,Y)[:2]
#     k=np.exp(param[0]); n=-1*param[1]; m=param[2]
#     R2 = 1 - residual / (Y.size * Y.var())
    return gh
