from pymongo import MongoClient
from utils import confLoad
import datetime
import traceback

class MongoHelper:
    def __init__(self):
        try:
            if confLoad.get('mongodb','DB_MONGO_USERNAME') is '':
                self.client = MongoClient(confLoad.get('mongodb','DB_MONGO_HOST'),int(confLoad.get('mongodb', 'DB_MONGO_PORT')))
            else:
                uri = 'mongodb://' + confLoad.get('mongodb', 'DB_MONGO_USERNAME') + ':' \
                      + confLoad.get('mongodb', 'DB_MONGO_PASSWORD') + '@' \
                      + confLoad.get('mongodb', 'DB_MONGO_HOST') + ':' \
                      + confLoad.get('mongodb', 'DB_MONGO_PORT')
                self.client = MongoClient(uri)
            self.db = self.client.get_database(confLoad.get('mongodb','DB_MONGO_DATABASE'))
        except:
            print(traceback.format_exc())

    def findOne(self, collectionName, filter=None):
        collection = self.db.get_collection(collectionName)
        return collection.find_one(filter)

    def findList(self, collectionName, filter=None):
        collection = self.db.get_collection(collectionName)
        return collection.find(filter)

    def updateOne(self,collectionName,update,filter=None):
        collection = self.db.get_collection(collectionName)
        if '$set' not in update:
            update['$set'] = {}
        update['$set']['updated_at'] = datetime.datetime.now()
        collection.update_one(filter,update)

    def insertOne(self,collectionName,document):
        collection = self.db.get_collection(collectionName)
        document['created_at'] = datetime.datetime.now()
        document['updated_at'] = datetime.datetime.now()
        collection.insert_one(document)

    def updateMany(self, collectionName, update, filter=None):
        collection = self.db.get_collection(collectionName)
        if '$set' not in update:
            update['$set'] = {}
        update['$set']['updated_at'] = datetime.datetime.now()
        collection.update_many(filter, update)

    def getSize(self,collectionName,filter=None):
        collection = self.db.get_collection(collectionName)
        return collection.count(filter)

    def closeClient(self):
        self.client.close()

