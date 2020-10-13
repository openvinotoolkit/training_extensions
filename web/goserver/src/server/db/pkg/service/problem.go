package service

import (
	"context"
	"log"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"

	n "server/common/names"
	t "server/db/pkg/types"
)

type ProblemFindOneRequestData struct {
	Id    primitive.ObjectID `bson:"_id" json:"id"`
	Class string             `bson:"class" json:"class"`
	Type  string             `bson:"type" json:"type"`
	Title string             `bson:"title" json:"title"`
}

func (s *basicDatabaseService) ProblemFindOne(ctx context.Context, req ProblemFindOneRequestData) (result t.Problem) {
	problemCollection := s.db.Collection(n.CProblem)
	filter := make(bson.M)
	if !primitive.ObjectID.IsZero(req.Id) {
		filter["_id"] = req.Id
	} else {
		if req.Title != "" {
			filter["title"] = req.Title
		}
		if req.Type != "" {
			filter["type"] = req.Type
		}
		if req.Class != "" {
			filter["class"] = req.Class
		}
	}
	err := problemCollection.FindOne(ctx, filter).Decode(&result)
	if err != nil {
		log.Println("ProblemFindOne.FindOne.Decode(&result)", err)
	}
	return
}

type ProblemFindRequestData struct {
	Page int64 `json:"page" bson:"page"`
	Size int64 `json:"size" bson:"size"`
}

func (s *basicDatabaseService) ProblemFind(ctx context.Context, req ProblemFindRequestData) (result t.ProblemFindResponse) {
	problemCollection := s.db.Collection(n.CProblem)
	option := options.Find()
	option.SetSkip(req.Size * (req.Page - 1))
	option.SetLimit(req.Size)
	total, err := problemCollection.EstimatedDocumentCount(ctx)
	if err != nil {
		return t.ProblemFindResponse{BaseList: t.BaseList{}}
	}
	cur, err := problemCollection.Find(ctx, bson.M{}, option)
	var items []t.Problem
	items = []t.Problem{}
	if err != nil {
		return t.ProblemFindResponse{BaseList: t.BaseList{Total: total}, Items: items}
	}

	for cur.Next(context.TODO()) {
		// create a value into which the single document can be decoded
		var elem t.Problem
		err := cur.Decode(&elem)
		if err != nil {
			return t.ProblemFindResponse{BaseList: t.BaseList{}}
		}

		items = append(items, elem)
	}

	if err := cur.Err(); err != nil {
		return t.ProblemFindResponse{BaseList: t.BaseList{}}
	}

	// Close the cursor once finished
	cur.Close(context.TODO())

	return t.ProblemFindResponse{BaseList: t.BaseList{Total: total}, Items: items}
}

type ProblemUpdateUpsertRequestData = t.ProblemWithouId

func (s *basicDatabaseService) ProblemUpdateUpsert(ctx context.Context, req ProblemUpdateUpsertRequestData) (result t.Problem) {
	problemCollection := s.db.Collection(n.CProblem)
	option := options.Update()
	option.SetUpsert(true)
	_, err := problemCollection.UpdateOne(ctx, bson.M{"title": req.Title}, bson.D{{"$set", req}}, option)
	if err != nil {
		log.Println("UpdateOne", err)
	}
	err = problemCollection.FindOne(ctx, bson.M{"title": req.Title}).Decode(&result)
	if err != nil {
		log.Println("ProblemUpdateUpsert.FindOne.Decode(&result)", err)
	}
	return result
}

type ProblemInsertOneRequestData = t.ProblemWithouId

func (s *basicDatabaseService) ProblemInsertOne(ctx context.Context, req ProblemInsertOneRequestData) (result t.Problem, err error) {
	problemCollection := s.db.Collection(n.CProblem)
	r, err := problemCollection.InsertOne(ctx, req)
	if err != nil {
		log.Println("ProblemInsertOne.InsertOne", err)
		return result, err
	}
	err = problemCollection.FindOne(ctx, bson.M{"_id": r.InsertedID}).Decode(&result)
	return result, err
}
