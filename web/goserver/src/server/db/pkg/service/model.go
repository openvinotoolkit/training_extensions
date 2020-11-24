package service

import (
	"context"
	"fmt"
	"log"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"

	n "server/common/names"
	t "server/db/pkg/types"
)

type ModelFindOneRequestData struct {
	Id primitive.ObjectID ` bson:"_id" json:"id"`
}

func (s *basicDatabaseService) ModelFindOne(ctx context.Context, req ModelFindOneRequestData) (result t.Model) {
	modelCollection := s.db.Collection(n.CModel)
	modelCollection.FindOne(ctx, bson.M{"_id": req.Id}).Decode(&result)
	fmt.Println("Model FindOne", result)
	return
}

type ModelFindRequestData struct {
	Page      int64              `bson:"page" json:"page"`
	Size      int64              `bson:"size" json:"size"`
	ProblemId primitive.ObjectID `bson:"problemId" json:"problemId"`
}

func (s *basicDatabaseService) ModelFind(ctx context.Context, req ModelFindRequestData) (result t.ModelFindResponse) {
	c := s.db.Collection(n.CModel)
	option := options.Find()
	option.SetSkip(req.Size * (req.Page - 1))
	option.SetLimit(req.Size)
	filter := bson.M{"problemId": req.ProblemId}
	total, err := c.CountDocuments(ctx, filter, options.Count())
	if err != nil {
		return t.ModelFindResponse{BaseList: t.BaseList{}}
	}
	cur, err := c.Find(ctx, filter, option)

	var items []t.Model
	if err != nil {
		return t.ModelFindResponse{BaseList: t.BaseList{Total: total}, Items: items}
	}
	for cur.Next(context.TODO()) {
		// create a value into which the single document can be decoded
		var elem t.Model
		err := cur.Decode(&elem)
		if err != nil {
			return t.ModelFindResponse{BaseList: t.BaseList{}}
		}
		items = append(items, elem)
	}

	if err := cur.Err(); err != nil {
		return t.ModelFindResponse{BaseList: t.BaseList{}}
	}

	// Close the cursor once finished
	cur.Close(context.TODO())
	log.Println(items)

	return t.ModelFindResponse{BaseList: t.BaseList{Total: total}, Items: items}
}

type ModelInsertOneRequestData = t.ModelWithoutId

func (s *basicDatabaseService) ModelInsertOne(ctx context.Context, req ModelInsertOneRequestData) (result t.Model, err error) {
	modelCollection := s.db.Collection(n.CModel)
	r, err := modelCollection.InsertOne(ctx, req)
	if err != nil {
		log.Println("ModelInsertOne.InsertOne", err)
		return result, err
	}
	err = modelCollection.FindOne(ctx, bson.M{"_id": r.InsertedID}).Decode(&result)
	return result, err
}

type ModelUpdateOneRequestData = t.Model

func (s *basicDatabaseService) ModelUpdateOne(ctx context.Context, req ModelUpdateOneRequestData) (result t.Model) {
	log.Println("Model Update One", req)
	modelCollection := s.db.Collection(n.CModel)
	r, err := modelCollection.UpdateOne(ctx, bson.M{"_id": req.Id}, bson.D{{"$set", req}})
	log.Println("Model Update One response", r)

	if err != nil {
		log.Println("ModelUpdateOne.UpdateOne", err)
		return result
	}
	log.Println("Model Update One response", req.Id)
	err = modelCollection.FindOne(ctx, bson.M{"_id": req.Id}).Decode(&result)
	if err != nil {
		log.Println("ModelUpdateOne.FindOne", err)
		return result
	}
	return

}

type ModelUpdateUpsertRequestData = t.ModelWithoutId

func (s *basicDatabaseService) ModelUpdateUpsert(ctx context.Context, req ModelUpdateUpsertRequestData) (result t.Model) {
	modelCollection := s.db.Collection(n.CModel)
	option := options.Update()
	option.SetUpsert(true)
	_, err := modelCollection.UpdateOne(ctx, bson.M{"name": req.Name, "problemId": req.ProblemId}, bson.D{{"$set", req}}, option)
	if err != nil {
		log.Println("UpdateOne", err)
	}
	err = modelCollection.FindOne(ctx, bson.M{"name": req.Name, "problemId": req.ProblemId}).Decode(&result)
	if err != nil {
		log.Println("ModelUpdateUpsert.FindOne", err)
	}
	return result
}

type ModelDeleteRequestData struct {
	Id primitive.ObjectID `json:"id" bson:"_id"`
}

type ModelDeleteResponseData struct {
	Id primitive.ObjectID `json:"id" bson:"_id"`
}

func (s *basicDatabaseService) ModelDelete(ctx context.Context, req ModelDeleteRequestData) ModelDeleteResponseData {
	modelCollection := s.db.Collection(n.CModel)
	if primitive.ObjectID.IsZero(req.Id) {
		return ModelDeleteResponseData{Id: primitive.NilObjectID}
	}
	filter := bson.M{"_id": req.Id}
	_, err := modelCollection.DeleteOne(ctx, filter)
	if err != nil {
		log.Println("ModelDelete.DeleteOne", err)
		return ModelDeleteResponseData{Id: primitive.NilObjectID}
	}
	return ModelDeleteResponseData{Id: req.Id}
}
