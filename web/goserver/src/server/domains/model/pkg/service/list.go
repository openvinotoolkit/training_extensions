package service

import (
	"context"

	"go.mongodb.org/mongo-driver/bson/primitive"

	modelFind "server/db/pkg/handler/model/find"
	kitendpoint "server/kit/endpoint"
)

type ListRequestData struct {
	Page      int64              `json:"page"`
	Size      int64              `json:"size"`
	ProblemId primitive.ObjectID `json:"problemId"`
}

func (s *basicModelService) List(
	ctx context.Context,
	req ListRequestData,
) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	respChan := modelFind.Send(
		ctx,
		s.Conn,
		modelFind.RequestData{
			Page:      req.Page,
			Size:      req.Size,
			ProblemId: req.ProblemId,
		},
	)
	go func() {
		defer close(returnChan)
		for r := range respChan {
			returnChan <- r
			if r.IsLast {
				return
			}
		}
	}()
	return returnChan
}
