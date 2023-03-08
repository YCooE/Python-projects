def estimatePi ( n ):
    s = 0
    for i in range(n):
        (x, y) = ( np.random.uniform(-1, 1), np.random.uniform(-1, 1) )
        if (x**2 + y**2 ) <= 1:
            s += 1
    return (s / n) * 4