import psycopg2

class DBInterface:

    def __init__(self, dbname=None, user=None):
        """ Initiate Matcher with dbname and username """

        if dbname and user:
            self.conn = self.connect(dbname, user)
            self.cursor = self.open_cursor()

    def connect(self, dbname, user):
        """ Create connection to db """

        self.dbname = dbname
        self.user = user

        return psycopg2.connect(dbname=dbname, user=user)

    def open_cursor(self):
        """ Create cursor """

        return self.conn.cursor()

    def execute(self, cmd):
        """ Execute command """

        self.cursor.execute(cmd)

    def fetchall(self, cmd):
        """ Fetch all values created by command """

        self.execute(cmd=cmd)
        return self.cursor.fetchall()

    def fetchone(self, cmd):
        """ Yield one value created by command"""

        self.execute(cmd)
        yield self.cursor.fetchone()

    def fetchmany(self, cmd, size):
        """ Yield 'size' values created by command """

        self.execute(cmd)
        yield self.cursor.fetchmany(size)