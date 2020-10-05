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

type CvatTaskFindRequestData struct {
	ProblemId primitive.ObjectID   `bson:"problemId" json:"problemId"`
	AssetIds  []primitive.ObjectID `bson:"assetIds" json:"assetIds"`
}

type CvatTaskFindResponse struct {
	Items []t.CvatTask `bson:"items" json:"items"`
}

func (s *basicDatabaseService) CvatTaskFind(ctx context.Context, req CvatTaskFindRequestData) (result CvatTaskFindResponse, err error) {
	cvatTaskCollection := s.db.Collection(n.CCvatTask)
	option := options.Find()
	cur, err := cvatTaskCollection.Find(
		ctx,
		bson.M{
			"problemId": req.ProblemId,
			"assetId":   bson.D{{"$in", req.AssetIds}},
		},
		option,
	)
	if err != nil {
		return CvatTaskFindResponse{Items: []t.CvatTask{}}, err
	}

	items := []t.CvatTask{}
	for cur.Next(context.TODO()) {
		// elem := t.CvatTask{}
		elem := new(t.CvatTask)
		elem2 := new(t.CvatTask)
		err := cur.Decode(elem)

		if err != nil {
			return CvatTaskFindResponse{Items: []t.CvatTask{}}, err
		}
		_ = bson.Unmarshal(cur.Current, elem2)
		items = append(items, *elem)
	}
	if err := cur.Err(); err != nil {
		return CvatTaskFindResponse{Items: items}, err
	}
	cur.Close(context.TODO())
	log.Println(items)

	return CvatTaskFindResponse{Items: items}, nil
}

type CvatTaskFindOneRequestData struct {
	Id primitive.ObjectID `bson:"_id" json:"id"`
}

func (s *basicDatabaseService) CvatTaskFindOne(ctx context.Context, req CvatTaskFindOneRequestData) (result t.CvatTask) {
	cvatTaskCollection := s.db.Collection(n.CCvatTask)
	err := cvatTaskCollection.FindOne(ctx, bson.M{"_id": req.Id}).Decode(&result)
	if err != nil {
		log.Print("CvatTaskFindOne", err)
	}
	return
}

type CvatTaskInsertOneRequestData struct {
	ProblemId primitive.ObjectID `bson:"problemId" json:"problemId"`
	AssetId   primitive.ObjectID `bson:"assetId" json:"assetId"`
	AssetPath string             `bson:"assetPath" json:"assetPath"`
	Status    string             `bson:"status" json:"status"`
	Params    t.CVATParams       `bson:"params" json:"params"`
	Progress  t.CvatTaskProgress `bson:"progress" json:"progress"`
}

func (s *basicDatabaseService) CvatTaskInsertOne(ctx context.Context, req CvatTaskInsertOneRequestData) (result t.CvatTask) {
	cvatTaskCollection := s.db.Collection(n.CCvatTask)
	insertRes, err := cvatTaskCollection.InsertOne(ctx, req)
	if err != nil {
		log.Print("CvatTaskInsertOne.InsertOne", err)
	}
	err = cvatTaskCollection.FindOne(ctx, bson.M{"_id": insertRes.InsertedID}).Decode(&result)
	if err != nil {
		log.Print("CvatTaskInsertOne.FindOne", err)
	}
	return
}

type CvatTaskUpdateOneRequestData = t.CvatTask

func (s *basicDatabaseService) CvatTaskUpdateOne(ctx context.Context, req CvatTaskUpdateOneRequestData) (result t.CvatTask) {
	cvatTaskCollection := s.db.Collection(n.CCvatTask)
	option := options.Update()
	_, err := cvatTaskCollection.UpdateOne(
		ctx,
		bson.M{"_id": req.Id},
		bson.M{"$set": req},
		option,
	)
	if err != nil {
		log.Print("CvatTaskUpdateOne.UpdateOne", err)
	}
	err = cvatTaskCollection.FindOne(ctx, bson.M{"_id": req.Id}).Decode(&result)
	if err != nil {
		log.Print("CvatTaskUpdateOne.FindOne", err)
	}
	return
}
