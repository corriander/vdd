import abc

# Python 2 & 3 compatible ABC superclass.
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})
