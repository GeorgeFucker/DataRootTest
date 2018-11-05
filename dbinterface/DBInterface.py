import psycopg2

class DBInterface:

    def __init__(self, dbname=None, user=None, password=None, host=None, port=5432):
        """ Initiate Matcher with dbname and username """

        self.conn = self.connect(dbname, user, password, host, port)
        self.cursor = self.open_cursor()

    def connect(self, dbname, user, password, host, port):
        """ Create connection to db """

        self.dbname = dbname
        self.user = user
        self.host = host
        self.port = port

        self.__password = password

        return psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

    def open_cursor(self):
        """ Create cursor """

        return self.conn.cursor()

    def execute(self, cmd):
        """ Execute command """

        self.cursor.execute(cmd)

    def fetchall(self):
        """ Fetch all values created by command """

        return self.cursor.fetchall()

    def fetchone(self):
        """ Yield one value created by command"""

        return self.cursor.fetchone()

    def fetchmany(self, size):
        """ Yield 'size' values created by command """

        return self.cursor.fetchmany(size)

    def commit(self):
        """ Commit changes in db """

        self.conn.commit()

    def disconnect(self):
        """ Close communication with the database """

        self.cursor.close()
        self.conn.close()