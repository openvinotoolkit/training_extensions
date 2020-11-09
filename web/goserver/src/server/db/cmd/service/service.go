package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"reflect"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	n "server/common/names"
	t "server/common/types"
	"server/db/pkg/endpoint"
	assetFind "server/db/pkg/handler/asset/find"
	assetFindOne "server/db/pkg/handler/asset/find_one"
	assetUpdateUpsert "server/db/pkg/handler/asset/update_upsert"
	buildFind "server/db/pkg/handler/build/find"
	buildFindOne "server/db/pkg/handler/build/find_one"
	buildInsertOne "server/db/pkg/handler/build/insert_one"
	buildUpdateOne "server/db/pkg/handler/build/update_one"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	cvatTaskFindOne "server/db/pkg/handler/cvat_task/find_one"
	cvatTaskInsertOne "server/db/pkg/handler/cvat_task/insert_one"
	cvatTaskUpdateOne "server/db/pkg/handler/cvat_task/update_one"
	modelDelete "server/db/pkg/handler/model/delete"
	modelFind "server/db/pkg/handler/model/find"
	modelFindOne "server/db/pkg/handler/model/find_one"
	modelInsertOne "server/db/pkg/handler/model/insert_one"
	modelUpdateOne "server/db/pkg/handler/model/update_one"
	modelUpdateUpsert "server/db/pkg/handler/model/update_upsert"
	problemDelete "server/db/pkg/handler/problem/delete"
	problemFind "server/db/pkg/handler/problem/find"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	problemUpdateUpsert "server/db/pkg/handler/problem/update_upsert"
	"server/db/pkg/service"
	longendpoint "server/kit/endpoint"
	kitutils "server/kit/utils"
)

func Run(serviceQueueName string, amqpAddr, amqpUser, amqpPass, mongoAddr *string) {
	ctx := context.Background()
	mongoUrl := fmt.Sprintf("mongodb://%s", *mongoAddr)
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoUrl))
	if err != nil {
		log.Panic(err)
	}
	amqpUrl := fmt.Sprintf("amqp://%s:%s@%s/", *amqpUser, *amqpPass, *amqpAddr)
	conn, err := rabbitmq.Dial(amqpUrl)
	if err != nil {
		log.Panic(err)
	}
	defer conn.Close()
	ch, err := conn.Channel()
	if err != nil {
		log.Panic(err)
	}
	defer ch.Close()
	servicesQueuesNames := []string{n.QAsset, n.QDatabase, n.QProblem, n.QTrainModel, n.QModel, n.QBuild}
	kitutils.AmqpServicesQueuesDelare(conn, servicesQueuesNames)

	db := client.Database("db")
	if err := initMongoIndexes(db); err != nil {
		log.Println("initMongoIndexes", err)

	}
	msgs, err := ch.Consume(
		serviceQueueName,
		"",
		true,
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		log.Println("Consume", serviceQueueName, err)
	}
	svc := service.New(db, getServiceMiddleware())
	eps := endpoint.New(svc)

	go func() {

		for msg := range msgs {
			var req t.AMQPRequestBody

			err := json.Unmarshal(msg.Body, &req)
			if err != nil {
				fmt.Println("Msg recieved", err)
				continue
			}
			fmt.Println(req.Request)
			switch req.Request {
			case assetFind.Request:
				go assetFind.Handle(eps, conn, msg)
			case assetFindOne.Request:
				go assetFindOne.Handle(eps, conn, msg)
			case assetUpdateUpsert.Request:
				go assetUpdateUpsert.Handle(eps, conn, msg)

			case buildFind.Request:
				go buildFind.Handle(eps, conn, msg)
			case buildFindOne.Request:
				go buildFindOne.Handle(eps, conn, msg)
			case buildInsertOne.Request:
				go buildInsertOne.Handle(eps, conn, msg)
			case buildUpdateOne.Request:
				buildUpdateOne.Handle(eps, conn, msg)

			case cvatTaskFind.Request:
				go cvatTaskFind.Handle(eps, conn, msg)
			case cvatTaskFindOne.Request:
				go cvatTaskFindOne.Handle(eps, conn, msg)
			case cvatTaskInsertOne.Request:
				go cvatTaskInsertOne.Handle(eps, conn, msg)
			case cvatTaskUpdateOne.Request:
				go cvatTaskUpdateOne.Handle(eps, conn, msg)

			case problemDelete.Request:
				go problemDelete.Handle(eps, conn, msg)
			case problemFind.Request:
				go problemFind.Handle(eps, conn, msg)
			case problemFindOne.Request:
				go problemFindOne.Handle(eps, conn, msg)
			case problemUpdateUpsert.Request:
				go problemUpdateUpsert.Handle(eps, conn, msg)

			case modelDelete.Request:
				go modelDelete.Handle(eps, conn, msg)
			case modelFind.Request:
				go modelFind.Handle(eps, conn, msg)
			case modelFindOne.Request:
				go modelFindOne.Handle(eps, conn, msg)
			case modelInsertOne.Request:
				go modelInsertOne.Handle(eps, conn, msg)
			case modelUpdateOne.Request:
				go modelUpdateOne.Handle(eps, conn, msg)
			case modelUpdateUpsert.Request:
				go modelUpdateUpsert.Handle(eps, conn, msg)
			default:
				log.Println("UNKNOWN REQUEST", req.Request)
			}
		}
	}()
	select {}
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Println("%s: %s", msg, err)
	}
}

func getServiceMiddleware() (mw []service.Middleware) {
	mw = []service.Middleware{}
	// Append your middleware here

	return
}

func getEndpointMiddleware() (mw map[string][]longendpoint.Middleware) {
	mw = map[string][]longendpoint.Middleware{}
	// Add you endpoint middleware here

	return
}

func initMongoIndexes(db *mongo.Database) error {
	if err := createBuildIndex(db); err != nil {
		return err
	}
	if err := createAssetIndex(db); err != nil {
		return err
	}
	if err := createModelIndex(db); err != nil {
		return err
	}
	return nil
}

func createBuildIndex(db *mongo.Database) error {
	indexes := mongo.IndexModel{
		Keys: bson.M{
			"problemId": 1,
			"name":      1,
		},
		Options: options.Index().SetUnique(true),
	}
	col := db.Collection(n.CBuild)
	ind, err := col.Indexes().CreateOne(context.TODO(), indexes)
	log.Println("CreateOne() index:", ind)
	log.Println("CreateOne() type:", reflect.TypeOf(ind), "\n")
	if err != nil {
		return err
	}
	return nil
}

func createAssetIndex(db *mongo.Database) error {
	indexes := mongo.IndexModel{
		Keys: bson.M{
			"name":         1,
			"parentFolder": 1,
		},
		Options: options.Index().SetUnique(true),
	}
	col := db.Collection(n.CAsset)
	ind, err := col.Indexes().CreateOne(context.TODO(), indexes)
	log.Println("CreateOne() index:", ind)
	log.Println("CreateOne() type:", reflect.TypeOf(ind), "\n")
	if err != nil {
		return err
	}
	return nil
}

func createModelIndex(db *mongo.Database) error {
	indexes := mongo.IndexModel{
		Keys: bson.M{
			"name":      1,
			"problemId": 1,
		},
		Options: options.Index().SetUnique(true),
	}
	col := db.Collection(n.CModel)
	ind, err := col.Indexes().CreateOne(context.TODO(), indexes)
	log.Println("CreateOne() index:", ind)
	log.Println("CreateOne() type:", reflect.TypeOf(ind), "\n")
	if err != nil {
		return err
	}
	return nil
}
