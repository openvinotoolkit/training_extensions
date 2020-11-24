package service

import (
	"context"

	"go.mongodb.org/mongo-driver/bson/primitive"

	modelDelete "server/db/pkg/handler/model/delete"
	kitendpoint "server/kit/endpoint"
)

type DeleteRequestData struct {
	Id primitive.ObjectID `bson:"_id" json:"id"`
}

func (s *basicModelService) Delete(ctx context.Context, req DeleteRequestData, responseChan chan kitendpoint.Response) {
	modelDeleteResp := <-modelDelete.Send(
		ctx,
		s.Conn,
		modelDelete.RequestData{
			Id: req.Id,
		},
	)
	responseChan <- modelDeleteResp
}
