package service

import (
	"context"

	"go.mongodb.org/mongo-driver/bson/primitive"

	problemDelete "server/db/pkg/handler/problem/delete"
	kitendpoint "server/kit/endpoint"
)

type DeleteRequestData struct {
	Id primitive.ObjectID `bson:"_id" json:"id"`
}

func (s *basicProblemService) Delete(ctx context.Context, req DeleteRequestData, responseChan chan kitendpoint.Response) {
	problemDeleteResp := <-problemDelete.Send(
		ctx,
		s.Conn,
		problemDelete.RequestData{
			Id: req.Id,
		},
	)
	responseChan <- problemDeleteResp
}
