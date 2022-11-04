{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    \"\"\"Entry point.\"\"\"\n",
    "    parser = argparse.ArgumentParser(\"MeanTeacher with CIFAR10.\")\n",
    "    parser.add_argument(\"--batch-size\", type=int, default=16)\n",
    "    parser.add_argument(\"--epochs\", type=int, default=20)\n",
    "    parser.add_argument(\"--learning-rate\", type=float, default=0.02)\n",
    "    parser.add_argument(\"--save-to\", type=str, default=\"model.pt\")\n",
    "    parser.add_argument(\"--dropout\", type=float, default=0.3)\n",
    "    parser.add_argument(\"--cuda\", action='store_true', default=False)\n",
    "    parser.add_argument(\"--supervised-ratio\", type=float, default=1.0)\n",
    "    parser.add_argument(\"--regularizer\", type=str, default=\"mt\")\n",
    "    parser.add_argument(\"--dataset\", type=str, default=\"CIFAR10\")\n",
    "    parser.add_argument(\"--ema-decay\", type=float, default=0.999)\n",
    "    parser.add_argument(\"--load\", type=str, default=None)\n",
    "    parser.add_argument(\"--test-only\", action='store_true')\n",
    "    parser.add_argument(\"--consistency-weight\", type=float, default=100)\n",
    "    parser.add_argument(\"--noise\", type=float, default=0.1)\n",
    "    parser.add_argument(\"--model\", type=str, default=\"wresnet\")\n",
    "    args = parser.parse_args()\n",
    "\n",
    "    device = 'cuda' if args.cuda else 'cpu'\n",
    "    model = create_model(args.model, args).to(device)\n",
    "    print(model)\n",
    "\n",
    "    if args.load:\n",
    "        model.load_state_dict(torch.load(args.load))\n",
    "\n",
    "    train_loader, val_loader = create_loaders_for_model(args.model, args)\n",
    "\n",
    "    criterion = nn.CrossEntropyLoss(ignore_index=-1)\n",
    "    optimizer = optim.SGD(model.parameters(),\n",
    "                          args.learning_rate,\n",
    "                          weight_decay=2e-4,\n",
    "                          nesterov=True,\n",
    "                          momentum=0.9)\n",
    "    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,\n",
    "                                                     len(train_loader) * (args.epochs + 50),\n",
    "                                                     eta_min=0,\n",
    "                                                     last_epoch=-1)\n",
    "    regularizer = MeanTeacherConsistencyCostRegularizer(disable_grad(copy.deepcopy(model)),\n",
    "                                                        args.ema_decay) if args.regularizer == \"mt\" else NullRegularizer()\n",
    "\n",
    "    training_loop(model,\n",
    "                  train_loader,\n",
    "                  val_loader,\n",
    "                  criterion,\n",
    "                  optimizer,\n",
    "                  scheduler,\n",
    "                  regularizer,\n",
    "                  device,\n",
    "                  args.epochs,\n",
    "                  lambda epoch: (1.0 - np.exp(-25.0 * np.square((epoch + 1) / args.epochs))) * args.consistency_weight,\n",
    "                  args.noise,\n",
    "                  args.test_only,\n",
    "                  result_writer(args.save_to))\n",
    "\n",
    "    if args.save_to:\n",
    "        torch.save(getattr(regularizer, \"teacher\", model).state_dict(), args.save_to)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10.2 64-bit",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.2"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "951cb00ebd66971857c5880c838618f69a707a5e4d3bee9fba08c4850d7bcc87"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
