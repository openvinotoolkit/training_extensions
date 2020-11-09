package service

import (
	"context"

	kitendpoint "server/kit/endpoint"
)

type ProblemsWatcherRequestData struct {
}

func (s *basicDatabaseWatcherService) Problem(ctx context.Context, req ProblemsWatcherRequestData, responseChan chan kitendpoint.Response) {

}
