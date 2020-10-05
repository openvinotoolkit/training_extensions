package service

import (
	"context"
	"log"

	"go.mongodb.org/mongo-driver/bson/primitive"

	problemFindOne "server/db/pkg/handler/problem/find_one"
	kitendpoint "server/kit/endpoint"
)

type DetailsRequestData struct {
	Id string `json:"id"`
}

func (s *basicProblemService) Details(ctx context.Context, req DetailsRequestData, responseChan chan kitendpoint.Response) {
	reqId, err := primitive.ObjectIDFromHex(req.Id)
	if err != nil {
		log.Fatal("ObjectIDFromHex")
	}
	problemFindOneRespChan := problemFindOne.Send(
		ctx,
		s.Conn,
		problemFindOne.RequestData{
			Id: reqId,
		},
	)
	for r := range problemFindOneRespChan {
		responseChan <- r
		if r.IsLast {
			return
		}
	}
}
